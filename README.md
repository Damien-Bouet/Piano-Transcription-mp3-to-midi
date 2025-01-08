# Piano Automatic Transcription : From mp3 sound to notes partitionPiano music

This project aims to explore the ways to convert mp3 solo music into midi file (which can be converted into music partition sheets with traditional softwares).

![Schema of architecture](img/schema.png)

---
# What it is doing :

## 1. Creation and augmentation of the dataset
I used in this project a dataset of simple piano songs which can be found on ![Kaggle](https://www.kaggle.com/datasets/kamilain/simple-midis).
The dataset is augmented by shifting the songs toward higher or lower octaves.

I used the ![Py-MeltySynth](https://github.com/sinshu/py-meltysynth) package to load and handle the midi files.

## 2. Convertion of the time series into the frequential domain using Short Time Fourier Transform (STFT)
I used `scipy.signal.ShortTimeFFT` to compute the STFTs.

## 3. Training a `pytorch` CNN+Dense network to predict the notes
<img src="img/model.png" alt="Model" style="width:70%;">

## 4. Example of test prediction
<img src="img/test_prediction.png" alt="Test prediction" style="width:70%;">
