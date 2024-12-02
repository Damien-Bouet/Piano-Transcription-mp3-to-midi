from typing import Tuple, Union

import librosa
from numpy.typing import NDArray
import meltysynth as ms


from IPython.display import Audio
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy.signal import get_window, ShortTimeFFT
from scipy.signal.windows import gaussian
import numpy as np

from display import midi_to_image


def mp3_to_ts(filename: str, sr=None, duration=None) -> Tuple[NDArray, float]:
    return librosa.load(filename, sr=sr, duration=duration)


def midi_to_wav(midi_file, output_file=None, duration=None):
    ms.midi_to_wav(midi_file, output_file=output_file, duration=duration)


def get_ft(t, sr) -> Tuple[NDArray, NDArray]:
    fourier = abs(rfft(t)) ** 2
    freqs = rfftfreq(n=len(t), d=1.0 / sr)
    return freqs, fourier


def get_frequency(notes: Union[int, NDArray]):
    return 440*(2**((notes-49)/12))


def convert_mp3_and_midi(mp3_file, midi_file, sr=22050, duration=5, ticks_per_beat=16, bpm=120, overlap_ratio=0.5, freq_resolution=0.5):
    ts, sr = mp3_to_ts(mp3_file, sr, duration=duration)
    
    midi = midi_to_image(midi_file, ticks_per_beat, bpm, duration)
    
    # Original parameters
    hop = int(sr / (bpm * ticks_per_beat/60))
    nperseg = int(hop / (1 - overlap_ratio))
    
    # win = get_window('hann', nperseg)
    # win = gaussian(nperseg, std=nperseg//10, sym=True)
    win = np.ones(nperseg)  # Rectangle window (flat)

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