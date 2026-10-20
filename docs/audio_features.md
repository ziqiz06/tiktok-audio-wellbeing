# Audio Feature Extraction Documentation

## Unit of Analysis
- Audio is extracted from each unique video URL.
- Features are computed once per video.
- Feature vectors are merged back onto repeated viewing events using `url`.

## General Processing
- Audio was resampled to 22,050 Hz to standardize processing across videos and reduce computational cost.
- Most audio features were computed at the frame level using Librosa.
- Frame-level features were aggregated to the video level using:
  - **Mean** (average level)
  - **Standard deviation** (variability)
  - **Volatility** (standard deviation of frame-to-frame differences, capturing abrupt changes over time)
- Tempo and beat count were retained as clip-level features.
- A non-silence ratio was computed to capture the proportion of active audio.

---

## Features

### RMS Energy
**Extraction:** `librosa.feature.rms(y=y)`  
**Level:** frame-level  
**Aggregation:** mean, standard deviation, volatility  
**Final variables:** `rms_1_mean`, `rms_1_std`, `rms_1_volatility`  
**Interpretation:** average loudness and its variability over time  

---

### Zero-Crossing Rate (ZCR)
**Extraction:** `librosa.feature.zero_crossing_rate(y=y)`  
**Aggregation:** mean, standard deviation, volatility  
**Final variables:** `zcr_1_mean`, `zcr_1_std`, `zcr_1_volatility`  
**Interpretation:** noisiness / percussive content and its variability  

---

### Spectral Centroid
**Extraction:** `librosa.feature.spectral_centroid(y=y, sr=sr)`  
**Aggregation:** mean, standard deviation, volatility  
**Final variables:** `centroid_1_mean`, `centroid_1_std`, `centroid_1_volatility`  
**Interpretation:** perceived brightness of the sound  

---

### Spectral Rolloff
**Extraction:** `librosa.feature.spectral_rolloff(y=y, sr=sr)`  
**Aggregation:** mean, standard deviation, volatility  
**Final variables:** `rolloff_1_mean`, `rolloff_1_std`, `rolloff_1_volatility`  
**Interpretation:** frequency below which ~85% of spectral energy lies  

---

### Spectral Bandwidth
**Extraction:** `librosa.feature.spectral_bandwidth(y=y, sr=sr)`  
**Aggregation:** mean, standard deviation, volatility  
**Final variables:** `bandwidth_1_mean`, `bandwidth_1_std`, `bandwidth_1_volatility`  
**Interpretation:** spread of frequencies around the spectral centroid  

---

### Spectral Contrast
**Extraction:** `librosa.feature.spectral_contrast(y=y, sr=sr)`  
**Level:** multiple frequency bands per frame  
**Aggregation:** mean, standard deviation, volatility per band  
**Final variables:** `contrast_1_*` … `contrast_7_*`  
**Interpretation:** difference between peaks and valleys in the frequency spectrum  

---

### Tempo (BPM)
**Extraction:** `librosa.beat.beat_track(y=y, sr=sr)`  
**Aggregation:** direct scalar output  
**Final variable:** `tempo`  
**Interpretation:** overall speed of the audio  

---

### Beat Structure
**Extraction:** `librosa.beat.beat_track(y=y, sr=sr)`  
**Aggregation:** count of detected beats  
**Final variable:** `beat_count`  
**Interpretation:** rhythmic density of the audio  

---

### Non-Silence Ratio
**Extraction:** `librosa.effects.split(y)`  
**Aggregation:** proportion of non-silent samples  
**Final variable:** `non_silence_ratio`  
**Interpretation:** proportion of the audio containing active sound  

---

### MFCCs (Mel-Frequency Cepstral Coefficients)
**Extraction:**  
`mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)`

**Level:**  
13 coefficients × time frames  

**Aggregation:**  
Mean, standard deviation, and volatility for each coefficient  

**Final variables:**  
`mfcc_1_mean` … `mfcc_13_mean`  
`mfcc_1_std` … `mfcc_13_std`  
`mfcc_1_volatility` … `mfcc_13_volatility`  

**Interpretation:**  
MFCCs capture the timbral characteristics (spectral shape) of the audio signal.  
Aggregating each coefficient provides a global representation of the clip’s sound profile and its temporal dynamics.  

---

### Chroma Features
**Extraction:** `librosa.feature.chroma_stft(y=y, sr=sr)`  
**Level:** 12 pitch classes per frame  
**Aggregation:** mean, standard deviation, volatility per bin  
**Final variables:** `chroma_1_*` … `chroma_12_*`  
**Interpretation:** distribution and variation of musical pitch classes  

---

### Tonnetz
**Extraction:**  
- `y_harmonic = librosa.effects.harmonic(y)`  
- `librosa.feature.tonnetz(y=y_harmonic, sr=sr)`

**Level:** 6 harmonic dimensions per frame  
**Aggregation:** mean, standard deviation, volatility  
**Final variables:** `tonnetz_1_*` … `tonnetz_6_*`  
**Interpretation:** harmonic relationships (e.g., major/minor tonal structure)  

---

## Summary Rule

Frame-level features are aggregated to the video level using mean, standard deviation, and volatility, producing one feature vector per video that captures both overall levels and temporal dynamics of the audio signal.