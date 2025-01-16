from typing import Tuple, Union

import librosa
from numpy.typing import NDArray
import scr.meltysynth as ms


from numpy.fft import rfft, rfftfreq
from scipy.signal import ShortTimeFFT, get_window
import numpy as np


import torch
import torch.nn.functional as F
from typing import List

from scr.display import midi_to_image


def mp3_to_ts(filename: str, sr=None, duration=None) -> Tuple[NDArray, float]:
    return librosa.load(filename, sr=sr, duration=duration)


def midi_to_wav(midi_file, output_file=None, duration=None, shift=0, fix_speed=None):
    return ms.midi_to_wav(midi_file, output_file=output_file, duration=duration, shift=shift, fix_speed=fix_speed)


def get_ft(t, sr) -> Tuple[NDArray, NDArray]:
    fourier = abs(rfft(t)) ** 2
    freqs = rfftfreq(n=len(t), d=1.0 / sr)
    return freqs, fourier


def get_frequency(notes: Union[int, NDArray]):
    return 440*(2**((notes-49)/12))


def create_dataset_with_stft(mp3_file, midi_file, midi_shift=0, sr=22050, duration=5, ticks_per_beat=16, bpm=120, overlap_ratio=0.5, freq_resolution=0.5, fix_velocity=False):
    ts, sr = mp3_to_ts(mp3_file, sr, duration=duration)
    
    midi = midi_to_image(midi_file, ticks_per_beat, bpm, duration, shift=midi_shift)

    # Original parameters
    hop = int(sr / (bpm * ticks_per_beat/60))
    nperseg = int(hop / (1 - overlap_ratio))   # 5512
    
    win = get_window('hann', nperseg)
    # win = gaussian(nperseg, std=nperseg//10, sym=True)
    # win = np.ones(nperseg)  # Rectangle window (flat)

    # Create ShortTimeFFT object
    SFT = ShortTimeFFT(win, hop, fs=sr, mfft=int(sr/freq_resolution))
    Sx = SFT.stft(ts)
    fs = SFT.f  # Frequency bins (Hz)
    t = SFT.t(ts.shape[0])  # Time bins (seconds)

    Zxx = np.abs(Sx)
    Zxx = Zxx.T
    if Zxx.shape[0] > midi.shape[0]:
        Zxx = Zxx[1:, :]
    if Zxx.shape[0] > midi.shape[0]:
        Zxx = Zxx[:-1, :]
    if Zxx.shape[0] > midi.shape[0]:
        raise RuntimeError("Error")
    fs = fs[1:-1]
    
    res = np.zeros((Zxx.shape[0], 88))
    for i, freq in enumerate(get_frequency(np.arange(1, 89))):        
        res[:, i] = Zxx[:, np.argmin(np.abs(fs-freq))]

    return res, midi



def harmonic_stacking(
    inputs: np.ndarray,
    harmonics: List[float],
    n_output_freqs: int
) -> torch.Tensor:
    """
    Harmonic stacking function.

    Args:
        x: Input tensor of shape (n_batch, n_times, n_freqs, 1).
        bins_per_semitone: The number of bins per semitone of the input CQT.
        harmonics: List of harmonics to use. Should be positive numbers.
        n_output_freqs: The number of frequency bins in each harmonic layer.

    Returns:
        Tensor of shape (n_batch, n_times, n_output_freqs, len(harmonics)).
    """
    x = torch.Tensor(inputs)
    assert x.ndim == 4, f"Expected input tensor to have 4 dimensions, got {x.ndim}."
    n_batch, n_times, n_freqs, _ = x.shape

    shifts = [
        int(torch.round(12.0 * torch.tensor(float(h))))
        for h in harmonics
    ]
    
    channels = []
    for shift in shifts:
        if shift == 0:
            padded = x
        elif shift > 0:
            # Shift right (pad at the end)
            paddings = (0, 0, 0, shift)  # Padding for (freq_start, freq_end)
            padded = F.pad(x[:, :, shift:, :], paddings, mode="constant", value=0)
        elif shift < 0:
            # Shift left (pad at the start)
            paddings = (0, 0, -shift, 0)  # Padding for (freq_start, freq_end)
            padded = F.pad(x[:, :, :shift, :], paddings, mode="constant", value=0)
        else:
            raise ValueError("Shift must be an integer.")

        # Ensure the padded tensor has exactly `n_freqs` frequency bins
        if padded.size(2) != n_freqs:
            padded = padded[:, :, :n_freqs, :]  # Truncate extra bins
        channels.append(padded)
    
    # Concatenate along the last dimension (harmonics channel)
    x = torch.cat(channels, dim=-1)
    # Truncate to the first n_output_freqs frequency channels
    x = x[:, :, :n_output_freqs, :]
    return x


def create_dataset_with_cqt(
    mp3_file,
    midi_file,
    midi_shift=0,
    sr=22050,
    duration=5,
    ticks_per_beat=16,
    bpm=120,
    harmonics: bool = False,
    use_batchnorm = True,
    sample_duration = None,
):
    
    # FFT_HOP = 256
    FFT_HOP = int(sr / (bpm * ticks_per_beat/60))
    CONTOURS_BINS_PER_SEMITONE = 1
    ANNOTATIONS_BASE_FREQUENCY = 27.5  # lowest key on a piano
    ANNOTATIONS_N_SEMITONES = 88  # number of piano keys
    AUDIO_SAMPLE_RATE = 22050    
    N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE
    
    MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))
    
    ts, sr = mp3_to_ts(mp3_file, sr, duration=duration)
    
    if sample_duration is not None:
        SLICES_LENGTH = AUDIO_SAMPLE_RATE * sample_duration
        N_FRAMES = int(sample_duration*AUDIO_SAMPLE_RATE/FFT_HOP)

        inputs = np.zeros((len(ts)//SLICES_LENGTH + 1, SLICES_LENGTH))
        for i in range(len(ts)//SLICES_LENGTH + 1):
            inputs[i, :min(len(ts) - SLICES_LENGTH*i, SLICES_LENGTH)] = ts[SLICES_LENGTH*i : min(len(ts), SLICES_LENGTH*(i+1))]
    else:
        inputs = ts[None, :]

    n_semitones=88

    x = np.abs(librosa.cqt(
        inputs,
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
    ))
    x = x[:, np.arange(0, x.shape[1], CONTOURS_BINS_PER_SEMITONE), :]
    
    # def to_log(inputs):
    #     """
    #     Takes an input with a shape of either (batch, x, y, z) or (batch, y, z)
    #     and rescales each (y, z) to dB, scaled 0 - 1.
    #     Only x=1 is supported.
    #     This layer adds 1e-10 to all values as a way to avoid NaN math.
    #     Output: (batch, y, z)
    #     """
    #     rank = len(inputs.shape)
    #     if rank == 4:
    #         assert inputs.shape[1] == 1, "If the rank is 4, the second dimension must be length 1"
    #         inputs = inputs[: ,0 ,: ,:]  # Remove the 'x' dimension

    #     elif rank != 3:
    #         raise ValueError(f"Only ranks 3 and 4 are supported! Received rank {rank} for {inputs.size()}.")

    #     # Convert magnitude to power
    #     power = inputs ** 2
    #     log_power = 10 * np.log10(power + 1e-10)

    #     # Min and max normalization
    #     log_power_min = log_power.min(axis=-1, keepdims=True).min(axis=-2, keepdims=True)
    #     log_power_offset = log_power - log_power_min
    #     log_power_offset_max = log_power_offset.max(axis=-1, keepdims=True).max(axis=-2, keepdims=True)
    #     log_power_normalized = log_power_offset / (log_power_offset_max + 1e-10)

    #     # Reshape back to the original shape
    #     return log_power_normalized


    # x = to_log(x)
    if use_batchnorm:
        if sample_duration is not None:
            x_mean = x.mean(axis=0, keepdims=True)  # Compute mean across batch
            x_std = x.std(axis=0, keepdims=True)    # Compute std deviation across batch
        else:
            x_mean = x.mean(axis=2, keepdims=True)  # Compute mean across batch
            x_std = x.std(axis=2, keepdims=True)    # Compute std deviation across batch
        x = (x - x_mean) / (x_std + 1e-10)

    # (n_batch, frequency, time, 1)
    
    if harmonics:
        x = x[:, :, :, None]
        x = np.transpose(x, (0, 2, 1, 3))
        
        res = harmonic_stacking(
            x,
            list(range(-2, 3)),
            88,
        ).numpy()
        
        res = np.transpose(res, (1, 0, 2, 3))
        n_batch, batch_length = res.shape[0], res.shape[1]
    else:
        res = x[0, :, :].T
        n_batch, batch_length = 1, res.shape[0]
        # res = harmonic_stacking(
        #     x,
        #     [1],
        #     N_FREQ_BINS_CONTOURS,
        # )

    midi = midi_to_image(midi_file, None, bpm, duration, shift=midi_shift, fix_output_length=n_batch*batch_length)
    
    if harmonics:
        y = np.zeros((n_batch, 88, batch_length))
        for i in range(n_batch):
            y[i, :, :min(len(midi) - batch_length*i, batch_length)] = midi[batch_length*i : min(len(midi), batch_length*(i+1)), :].T
    else:
        y = midi
    return np.array(res), y

