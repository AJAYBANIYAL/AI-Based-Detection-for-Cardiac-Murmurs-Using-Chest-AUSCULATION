# AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION
Sound Analysis of Heart Sound using Libosa , used 2DCNN  and  LSTM and Web based implementaiton at https://murmur.streamlit.app/

Thesis Project for M.tech in Cse(Artificial Intelligence) at CDAC, Mohali Under the guidance of Dr.Jaspal Singh

# IEEE Conference Research paper 80 - 120 days.
# For Future Collaboration
Contact Email : ajaybanyalofficial@gmail.com

# Introduction
Heart disease is one of the major causes of death with a rate of 17.9 million per year of which 85% of them are due to heart attack and stroke, with more than 75% of deaths occurring in Third world countries reasons being Obesity, Lifestyle change, Fatty foods & Lack of exercise.

At every heartbeat, blood flows through the chambers and valves, creating a characteristic "lub" or "dub" sound. When the blood flow through the heart is disrupted, it creates turbulence that produces an additional sound that is superimposed on the normal heart sounds and is referred to as murmur


![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/b289f356-3027-4c3a-b7f2-e54ace6342b0)


Murmurs are caused by a variety of underlying conditions, including valve defects, holes in the heart, or abnormal blood flow patterns. Depending on the location and cause of the murmur, it may be heard at different points during the cardiac cycle, such as during systole (when the heart is contracting) or diastole (when the heart is relaxing).


![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/fea64c65-41e1-4f81-b18b-6619f86bde18)

# Motivation

1.Early Detection of heart murmur in children, adults or old people can be life saving and cost saving or until we act after the symptoms start to show or its too late to be cured.  

2.With Artificial Intelligence Early detection of cardiac murmur is an absolute  life saving idea that points the future scope of 24x7 live monitoring system for heart sounds  attached near to the heart.

# Research Objectives

1.Study for various techniques for AI based audio analysis.

2.Development of AI based 2D CNN & LSTM model for detection of cardiac Murmur in heart sounds.

3.Comparative evaluation of the Developed Model’s.

4.Development of a web based implementation for a better performing model

# Dataset

1.The dataset used for this work was first produced for a machine learning task to categorize heartbeat Sounds. The dataset includes heartbeat audio recordings from two different sources from members of the general public using the digital stethoscope through iphone append  from a clinical experiment conducted in hospitals using the DigiScope digital stethoscope after the data is Augmented and length of the sound is decreased.Visualization of Dataset according to classes.
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/43da8079-64d4-4b1e-8ef8-01961e49db62)





2.The dataset can be found at https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds.

3.Three Generation Stethoscope
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/c99e1437-13df-4239-844d-e5345ae7df31)

# Data Preprocessing
1.For data augmentation murmur audio files were collected from different dataset to increase the murmur samples in dataset.
2.The dataset was augmented to 22 kHz, after which the audio files were truncated to 10 seconds and further sliced into 4-sec segments for both 2D-CNN and LSTM model.
3.A cardiac cycle containing “lub”, “dub” sound has a duration of 0.8 seconds and For 4-sec segments minimum four and  maximum five cardiac cycle can occur in those 5 seconds.
4.Used a audio filter bandpass for 20 -22khz ; Butterworth filter using scipy.

# Feature Extraction
# Librosa :
Librosa is a Python package for analyzing and processing audio signals, with a focus on music and sound data. It provides a collection of algorithms and tools for audio related task.
Spectral Features: Spectral features consist of variety of methods like spectral centroid, spectral rolloff, spectral flux, Mel-frequency cepstral coefficients (MFCCs)  and many more.
Onset Detection: Onset detection in librosa is performed using the onset_detect() function. This function takes an audio signal and returns a list of the detected onset times, in the specified units (by default, frames).
Beat and Tempo:The beat and tempo of an audio signal can be estimated using the beat.beat_track() function in librosa. This function takes an audio signal and returns a list of the estimated beat times, in the specified units (by default, seconds), and the estimated tempo in beats per minute (BPM).

![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/68045764-150c-4a37-9d78-0d06820bd1ea)

# MFCCS(Mel-frequency cepstral coefficients):

The process of extracting MFCC involves a series of stages that comprise pre-emphasis, framing, windowing, Fourier transform, mel frequency filtering, logarithmic compression, and discrete cosine transform. These stages are implemented to convert the unprocessed heart sound signal into a group of MFCC coefficients that can serve as features for subsequent analysis. In sound processing, the mel-frequency cepstrum (MFCC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.
The following is a typical derivation of MFCCs is:
Perform a signal's Fourier transform on a windowed extract.
Use triangle overlapping windows or, as an alternative, cosine overlapping windows map the powers of the spectrum acquired in step one onto the mel scale.
Calculate the power logs at each of the mel frequencies.
Treat the list of mel log powers' discrete cosine transform as though it were a signal.
The amplitudes of the resultant spectrum are the MFCCs.
# Mel Spectrogram :

A mel-spectrogram is a visual representation of the spectrum of frequencies in a sound, with the y-axis representing frequency and the x-axis representing time. The brightness of the image corresponds to the amplitude of the sound at each frequency. Mel spectrograms are generated by applying a Fourier transform to a signal and then mapping the resulting frequencies onto the mel scale, a logarithmic scale that is based on human perception of pitch.

![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/d4507a0e-b545-4f8e-8dd9-325b21763096)





# Melspectrogram+mfccs FeatureS for 2DCNN
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/77f14da5-9b34-4bf8-9652-51d416510d61)
# mfccs features for LSTM
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/d93b1393-9f6a-4d4a-93f4-a41e2018b75d)
# Train - Test Split
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/9dda4450-365f-4d5d-8383-1d83fdcc8fee)
# Model Architecture for 2DCNN
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/58b68fbd-0314-4cfe-a005-ef7a7663b642)

# Model Architecture for LSTM
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/3ac2b78b-63ec-49a9-aa2e-94a2326fc943)

# Results
# Precision, Recall, F1-Score.
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/603b213b-d37a-4580-aaa3-ba5618c8ef77)

# Accuracy.
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/b3f8ac41-5c0b-4f08-be2a-2122c3441161)

# confusion Matrix
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/12de0a58-28c3-4252-8b49-ab52d56b3ab2)
![image](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/7b553506-d22e-443b-9d85-ca83407c6dff)

# web based implemetation
Web based implementaiton at https://murmur.streamlit.app/
![Capture](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/d293a7ec-9c84-4608-91e2-a1a38d9499d4)
![normal hert sound](https://github.com/AJAYBANIYAL/AI-BASED-DETECTION-OF-CARDIAC-MURMUR-USING-CHEST-AUSCULATION-/assets/33643674/2c670a49-8015-441c-96e8-cbf044a0c7cd)

# Future Scope
1.The developed tool can be further improved by incorporating more advanced deep learning models and techniques.

2.The dataset used in this study can be expanded to include more diverse and larger samples for better generalization and accuracy.

3.The tool can be integrated with electronic health records and other healthcare technologies to provide a more comprehensive and efficient healthcare system.











