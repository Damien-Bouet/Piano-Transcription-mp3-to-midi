
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Optional, Tuple, Union

import scipy.signal



def next_power_of_2(A: int) -> int:
    """A helper function to calculate the next nearest number to the power of 2."""
    return int(np.ceil(np.log2(A)))


def early_downsample(
    sr: Union[float, int],
    hop_length: int,
    n_octaves: int,
    nyquist_hz: float,
    filter_cutoff_hz: float,
) -> Tuple[Union[float, int], int, int]:
    """Return new sampling rate and hop length after early downsampling"""
    downsample_count = early_downsample_count(nyquist_hz, filter_cutoff_hz, hop_length, n_octaves)
    downsample_factor = 2 ** (downsample_count)

    hop_length //= downsample_factor  # Getting new hop_length
    new_sr = sr / float(downsample_factor)  # Getting new sampling rate

    return new_sr, hop_length, downsample_factor


# The following two downsampling count functions are obtained from librosa CQT
# They are used to determine the number of pre resamplings if the starting and ending frequency
# are both in low frequency regions.
def early_downsample_count(nyquist_hz: float, filter_cutoff_hz: float, hop_length: int, n_octaves: int) -> int:
    """Compute the number of early downsampling operations"""

    downsample_count1 = max(0, int(np.ceil(np.log2(0.85 * nyquist_hz / filter_cutoff_hz)) - 1) - 1)
    num_twos = next_power_of_2(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)


def get_window_dispatch(window: Union[str, Tuple[str, float]], N: int, fftbins: bool = True) -> Optional[NDArray]:
    if isinstance(window, str):
        return scipy.signal.get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == "gaussian":
            assert window[1] >= 0
            sigma = np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))
            return scipy.signal.get_window(("gaussian", sigma), N, fftbins=fftbins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
            return None
    elif isinstance(window, float):
        Warning("You are using Kaiser window with beta factor " + str(window) + ". Correct behaviour not checked.")
        return None
    else:
        raise Exception("The function get_window from scipy only supports strings, tuples and floats.")


def create_cqt_kernels(
    Q: float,
    fs: float,
    fmin: float,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    norm: int = 1,
    window: str = "hann",
    fmax: Optional[float] = None,
    topbin_check: bool = True,
) -> Tuple[np.array, int, np.array, np.array]:
    """
    Automatically create CQT kernels in time domain
    """

    fftLen = 2 ** next_power_of_2(np.ceil(Q * fs / fmin))

    if (fmax is not None) and (n_bins is None):
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    else:
        warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    if np.max(freqs) > fs / 2 and topbin_check is True:
        raise ValueError(
            "The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins".format(np.max(freqs))
        )

    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

    lengths = np.ceil(Q * fs / freqs)
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        _l = np.ceil(Q * fs / freq)

        # Centering the kernels, pad more zeros on RHS
        start = int(np.ceil(fftLen / 2.0 - _l / 2.0)) - int(_l % 2)

        sig = (
            get_window_dispatch(window, int(_l), fftbins=True)
            * np.exp(np.r_[-_l // 2 : _l // 2] * 1j * 2 * np.pi * freq / fs)
            / _l
        )

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start : start + int(_l)] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start : start + int(_l)] = sig

    return tempKernel, fftLen, lengths, freqs


def pad_center(data: np.ndarray, size: int, axis: int = -1, **kwargs: Any) -> np.ndarray:
    """Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])
    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered
    size : int >= len(data) [scalar]
        Length to pad `data`
    axis : int
        Axis along which to pad and center the data
    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ValueError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(("Target size ({:d}) must be at least input size ({:d})").format(size, n))

    return np.pad(data, lengths, **kwargs)




def create_lowpass_filter(
    band_center: float = 0.5,
    kernel_length: int = 256,
    transition_bandwidth: float = 0.03,
    dtype: np.dtype = np.float32,
) -> NDArray:
    """
    Calculate the highest frequency we need to preserve and the lowest frequency we allow
    to pass through. Note that frequency is on a scale from 0 to 1 where 0 is 0 and 1 is
    the Nyquist frequency of the signal BEFORE downsampling.
    """

    passband_max = band_center / (1 + transition_bandwidth)
    stopband_min = band_center * (1 + transition_bandwidth)

    # We specify a list of key frequencies for which we will require
    # that the filter match a specific output gain.
    # From [0.0 to passband_max] is the frequency range we want to keep
    # untouched and [stopband_min, 1.0] is the range we want to remove
    key_frequencies = [0.0, passband_max, stopband_min, 1.0]

    # We specify a list of output gains to correspond to the key
    # frequencies listed above.
    # The first two gains are 1.0 because they correspond to the first
    # two key frequencies. the second two are 0.0 because they
    # correspond to the stopband frequencies
    gain_at_key_frequencies = [1.0, 1.0, 0.0, 0.0]

    # This command produces the filter kernel coefficients
    filter_kernel = scipy.signal.firwin2(kernel_length, key_frequencies, gain_at_key_frequencies)

    return np.array(filter_kernel, dtype=dtype)


def get_early_downsample_params(
    sr: Union[float, int],
    hop_length: int,
    fmax_t: float,
    Q: float,
    n_octaves: int,
    dtype: np.dtype,
) -> Tuple[Union[float, int], int, float, torch.Tensor, bool]:
    """Compute downsampling parameters used for early downsampling"""

    window_bandwidth = 1.5  # for hann window
    filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)
    
    sr, hop_length, downsample_factor = early_downsample(sr, hop_length, n_octaves, sr // 2, filter_cutoff)
    
    if downsample_factor != 1:
        earlydownsample = True
        early_downsample_filter = create_lowpass_filter(
            band_center=1 / downsample_factor,
            kernel_length=256,
            transition_bandwidth=0.03,
            dtype=dtype,
        )
    else:
        early_downsample_filter = None
        earlydownsample = False

    return sr, hop_length, downsample_factor, early_downsample_filter, earlydownsample


def get_cqt_complex(
    x: torch.Tensor,
    cqt_kernels_real: torch.Tensor,
    cqt_kernels_imag: torch.Tensor,
    hop_length: int,
    padding: nn.Module,
) -> torch.Tensor:
    """Multiplying the STFT result with the cqt_kernel"""
    
    try:
        x = padding(x)  # When center is True, we need padding at the beginning and ending
    except Exception:
        warnings.warn(
            f"\ninput size = {x.shape}\tkernel size = {cqt_kernels_real.shape[-1]}\n"
            "padding with reflection mode might not be the best choice, try using constant padding",
            UserWarning,
        )
        pad = cqt_kernels_real.shape[-1] // 2
        x = F.pad(x, (pad, pad), mode='reflect')

    # Perform convolution in PyTorch
    CQT_real = F.conv1d(
        x.transpose(1, 2), 
        cqt_kernels_real.transpose(0, 1), 
        stride=hop_length
    ).transpose(1, 2)
    
    CQT_imag = -F.conv1d(
        x.transpose(1, 2), 
        cqt_kernels_imag.transpose(0, 1), 
        stride=hop_length
    ).transpose(1, 2)

    return torch.stack((CQT_real, CQT_imag), dim=-1)


def downsampling_by_n(
    x: torch.Tensor, 
    filter_kernel: torch.Tensor, 
    n: float, 
    match_torch_exactly: bool = True
) -> torch.Tensor:
    """
    Downsample the given tensor using the given filter kernel.
    The input tensor is expected to have shape `(n_batches, channels, width)`,
    and the filter kernel is expected to have shape `(num_output_channels,)` (i.e.: 1D)
    """
    if match_torch_exactly:
        # Manual padding to match TensorFlow-like behavior
        pad = (filter_kernel.shape[-1] - 1) // 2
        padded = F.pad(x, (pad, pad), mode='reflect')
        
        # Perform convolution
        result = torch.nn.functional.conv1d(
            padded.transpose(1, 2), 
            filter_kernel[:, None, None], 
            stride=int(n)
        ).transpose(1, 2)
    else:
        # PyTorch's default convolution behavior
        result = torch.nn.functional.conv1d(
            x.transpose(1, 2), 
            filter_kernel[:, None, None], 
            stride=int(n), 
            padding='same'
        ).transpose(1, 2)

    return result


class ReflectionPad1D(nn.Module):
    """
    Replica of PyTorch's nn.ReflectionPad1D
    """
    def __init__(self, padding: Union[int, Tuple[int]] = 1):
        super().__init__()
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.padding, mode='reflect')


class ConstantPad1D(nn.Module):
    """
    Replica of PyTorch's nn.ConstantPad1D
    """
    def __init__(self, padding: Union[int, Tuple[int]] = 1, value: float = 0):
        super().__init__()
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.padding, mode='constant', value=self.value)



class CQT2010v2(nn.Module):
    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=32.70,
        fmax=None,
        n_bins=84,
        filter_scale=1,
        bins_per_octave=12,
        norm=True,
        basis_norm=1,
        window='hann',
        pad_mode='reflect',
        earlydownsample=True,
        trainable=False,
        output_format='Magnitude',
        match_torch_exactly=True
    ):
        super().__init__()
        
        self.sample_rate = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.n_bins = n_bins
        self.filter_scale = filter_scale
        self.bins_per_octave = bins_per_octave
        self.norm = norm
        self.basis_norm = basis_norm
        self.window = window
        self.pad_mode = pad_mode
        self.earlydownsample = earlydownsample
        self.trainable = trainable
        self.output_format = output_format
        self.match_torch_exactly = match_torch_exactly
        self.normalization_type = "librosa"
        
        # Placeholder for calculated attributes
        self.lowpass_filter = None
        self.downsample_factor = None
        self.early_downsample_filter = None
        self.n_octaves = None
        self.fmin_t = None
        self.basis = None
        self.cqt_kernels_real = None
        self.cqt_kernels_imag = None
        self.n_fft = None
        self.frequencies = None
        self.lengths = None

    def _get_input_shape(self, x):
        """Infer input shape and reshape accordingly."""
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        return x

    def build(self, input_shape=None):
        """
        Prepare CQT kernels and related parameters.
        This is equivalent to the TensorFlow 'build' method.
        """
        Q = float(self.filter_scale) / (2 ** (1 / self.bins_per_octave) - 1)

        self.lowpass_filter = torch.from_numpy(
            create_lowpass_filter(band_center=0.5, kernel_length=256, transition_bandwidth=0.001)
        ).float()

        # Calculate number of filters and octaves
        n_filters = min(self.bins_per_octave, self.n_bins)
        self.n_octaves = int(np.ceil(float(self.n_bins) / self.bins_per_octave))

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = self.fmin * 2 ** (self.n_octaves - 1)
        remainder = self.n_bins % self.bins_per_octave

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((self.bins_per_octave - 1) / self.bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / self.bins_per_octave)

        self.fmin_t = fmax_t / 2 ** (1 - 1 / self.bins_per_octave)  # Adjusting the top minimum bins
        if fmax_t > self.sample_rate / 2:
            raise ValueError(
                f"The top bin {fmax_t}Hz has exceeded the Nyquist frequency, please reduce the n_bins"
            )

        if self.earlydownsample:
            (
                self.sample_rate,
                self.hop_length,
                self.downsample_factor,
                self.early_downsample_filter,
                _,  # earlydownsample flag (ignored)
            ) = get_early_downsample_params(
                self.sample_rate, 
                self.hop_length, 
                fmax_t, 
                Q, 
                self.n_octaves, 
                np.float32
            )
        else:
            self.downsample_factor = 1.0
        
        # Preparing CQT kernels
        basis, self.n_fft, _, _ = create_cqt_kernels(
            Q,
            self.sample_rate,
            self.fmin_t,
            n_filters,
            self.bins_per_octave,
            norm=self.basis_norm,
            topbin_check=False
        )

        # Frequencies for all bins
        freqs = self.fmin * 2.0 ** (np.r_[0 : self.n_bins] / float(self.bins_per_octave))
        self.frequencies = freqs

        self.lengths = np.ceil(Q * self.sample_rate / freqs)

        self.basis = basis
        
        # Convert kernels to PyTorch tensors
        self.cqt_kernels_real = torch.from_numpy(basis.real.astype(np.float32)).unsqueeze(1)
        self.cqt_kernels_imag = torch.from_numpy(basis.imag.astype(np.float32)).unsqueeze(1)

        if self.trainable:
            self.cqt_kernels_real = nn.Parameter(self.cqt_kernels_real)
            self.cqt_kernels_imag = nn.Parameter(self.cqt_kernels_imag)

        return self

    def forward(self, x):
        # Reshape input
        x = self._get_input_shape(x)

        if self.earlydownsample:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor, self.match_torch_exactly)

        hop = self.hop_length

        # Getting the top octave CQT
        CQT = self._get_cqt_complex(x, self.cqt_kernels_real, self.cqt_kernels_imag, hop)

        x_down = x  # Preparing a new variable for downsampling

        for _ in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = downsampling_by_n(x_down, self.lowpass_filter, 2, self.match_torch_exactly)
            CQT1 = self._get_cqt_complex(x_down, self.cqt_kernels_real, self.cqt_kernels_imag, hop)
            CQT = torch.cat((CQT1, CQT), dim=1)

        CQT = CQT[:, -self.n_bins:, :]  # Removing unwanted bottom bins

        # Normalizing the output with the downsampling factor
        CQT = CQT * self.downsample_factor

        # Normalize again to get same result as librosa
        if self.normalization_type == "librosa":
            CQT *= torch.sqrt(torch.tensor(self.lengths.reshape((-1, 1, 1)), dtype=x.dtype))
        elif self.normalization_type == "convolutional":
            pass
        elif self.normalization_type == "wrap":
            CQT *= 2
        else:
            raise ValueError(f"The normalization_type {self.normalization_type} is not part of our current options.")

        # Output formatting
        if self.output_format.lower() == "magnitude":
            return torch.transpose(torch.sqrt(torch.sum(CQT**2, dim=-1)), 1, 2)
        elif self.output_format.lower() == "complex":
            return CQT
        elif self.output_format.lower() == "phase":
            phase = torch.atan2(CQT[..., 1], CQT[..., 0])
            phase_real = torch.cos(phase)
            phase_imag = torch.sin(phase)
            return torch.stack((phase_real, phase_imag), dim=-1)

    def _get_cqt_complex(self, x, kernels_real, kernels_imag, hop):
        """
        Compute complex CQT by performing convolution in frequency domain.
        """
        # Prepare FFT of input
        x_fft = torch.fft.rfft(x.squeeze(1), n=self.n_fft)
        
        # Complex kernel multiplication
        kernel_fft_real = torch.fft.rfft(kernels_real.squeeze(1), n=self.n_fft)
        kernel_fft_imag = torch.fft.rfft(kernels_imag.squeeze(1), n=self.n_fft)
        
        # Perform complex multiplication in frequency domain
        X_k_real = x_fft.real * kernel_fft_real.real - x_fft.imag * kernel_fft_real.imag
        X_k_imag = x_fft.real * kernel_fft_imag.real - x_fft.imag * kernel_fft_imag.imag
        
        # Take chunks based on hop length
        X_k_real = X_k_real[:, :, ::hop]
        X_k_imag = X_k_imag[:, :, ::hop]
        
        # Inverse FFT to get complex result
        CQT = torch.stack([X_k_real, X_k_imag], dim=-1)
        
        return CQT

    def extra_repr(self):
        """String representation of the layer for printing."""
        return (
            f"sample_rate={self.sample_rate}, "
            f"hop_length={self.hop_length}, "
            f"n_bins={self.n_bins}, "
            f"output_format={self.output_format}"
        )
    

CQT = CQT2010v2