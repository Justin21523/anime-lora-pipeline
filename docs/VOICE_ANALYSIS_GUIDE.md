# Voice Analysis & Characterization Guide
## Understanding and Analyzing Character Voices

---

## What Makes Each Voice Unique?

Every person's voice is unique due to multiple characteristics. Understanding these helps us:
1. **Identify speakers** (speaker diarization)
2. **Clone voices** (voice conversion)
3. **Analyze emotional states**
4. **Detect authenticity**

---

## Core Voice Characteristics

### 1. **Fundamental Frequency (F0) - 基頻/音高**

**What is it?**
- The lowest frequency of vocal cord vibration
- Perceived as voice "pitch"
- Measured in Hertz (Hz)

**Typical Ranges**:
```
Adult Male:     85-180 Hz   (低沉男聲)
Adult Female:   165-255 Hz  (女聲)
Child:          250-400 Hz  (童聲)
Anime Character Variations:
  - Young boy (Endou):      200-300 Hz
  - Teenage girl:           220-350 Hz
  - Deep male voice:        100-150 Hz
  - High-pitched character: 300-450 Hz
```

**Why it matters**:
- Primary identifier of speaker identity
- Changes with emotion (excitement → higher, sadness → lower)
- Can be visualized as pitch contour

**Extraction Methods**:
```python
import librosa
import numpy as np

def extract_f0(audio_path):
    """
    Extract fundamental frequency using multiple methods
    """
    y, sr = librosa.load(audio_path)

    # Method 1: CREPE (most accurate, AI-based)
    from crepe import predict
    time, frequency, confidence, activation = predict(
        y, sr, viterbi=True, step_size=10
    )

    # Method 2: pYIN (good balance)
    f0_pyin, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=80, fmax=600, sr=sr
    )

    # Method 3: Classic autocorrelation
    f0_auto = librosa.yin(y, fmin=80, fmax=600, sr=sr)

    return {
        'crepe': frequency,
        'pyin': f0_pyin,
        'autocorrelation': f0_auto,
        'mean_f0': np.nanmean(frequency),
        'std_f0': np.nanstd(frequency)
    }
```

---

### 2. **Formants - 共振峰**

**What are they?**
- Resonant frequencies of the vocal tract
- Determined by shape of mouth, tongue, throat
- Give voice its unique "color" or timbre

**Key Formants**:
```
F1 (First Formant):  250-950 Hz
  - Related to tongue height
  - Higher F1 = more open mouth (vowel 'a')

F2 (Second Formant): 850-2500 Hz
  - Related to tongue position (front/back)
  - Higher F2 = tongue forward (vowel 'i')

F3 (Third Formant):  1500-3500 Hz
  - Affects voice quality

F4, F5: Higher formants
  - Contribute to voice uniqueness
```

**Example Visualization**:
```
Vowel 'a' (あ):
  F1: ~700 Hz  (open mouth)
  F2: ~1200 Hz

Vowel 'i' (い):
  F1: ~300 Hz  (closed mouth)
  F2: ~2300 Hz (tongue forward)

Vowel 'u' (う):
  F1: ~300 Hz
  F2: ~800 Hz  (lips rounded, tongue back)
```

**Why it matters**:
- Formants are speaker-specific
- Even if two people have same pitch, formants differ
- Critical for voice cloning

**Extraction**:
```python
import parselmouth
from parselmouth.praat import call

def extract_formants(audio_path, max_formant=5500):
    """
    Extract formant frequencies using Praat

    Args:
        max_formant: 5500 for male, 5000 for female voices
    """
    sound = parselmouth.Sound(audio_path)

    # Create formant object
    formant = sound.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=max_formant
    )

    # Extract formant values over time
    times = formant.ts()
    formants = {f'F{i}': [] for i in range(1, 6)}

    for t in times:
        for i in range(1, 6):
            f_value = formant.get_value_at_time(i, t)
            formants[f'F{i}'].append(f_value)

    # Calculate statistics
    formant_stats = {}
    for i in range(1, 6):
        values = [v for v in formants[f'F{i}'] if not np.isnan(v)]
        formant_stats[f'F{i}_mean'] = np.mean(values) if values else 0
        formant_stats[f'F{i}_std'] = np.std(values) if values else 0

    return formant_stats
```

---

### 3. **Spectral Characteristics - 頻譜特徵**

#### A. **Spectral Centroid - 頻譜重心**
- "Center of mass" of the spectrum
- Higher = brighter, sharper voice
- Lower = darker, warmer voice

```python
def extract_spectral_features(audio_path):
    """Extract various spectral features"""
    y, sr = librosa.load(audio_path)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Spectral rolloff (frequency below which X% of energy lies)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Zero crossing rate (how often signal changes sign)
    zcr = librosa.feature.zero_crossing_rate(y)

    return {
        'spectral_centroid_mean': np.mean(centroid),
        'spectral_rolloff_mean': np.mean(rolloff),
        'spectral_bandwidth_mean': np.mean(bandwidth),
        'zero_crossing_rate_mean': np.mean(zcr)
    }
```

#### B. **MFCCs (Mel-Frequency Cepstral Coefficients)**
- Most important features for voice recognition
- Mimic human auditory perception
- 13-40 coefficients typically used

```python
def extract_mfcc(audio_path, n_mfcc=20):
    """
    Extract MFCC features

    MFCCs are the standard in:
      - Speaker identification
      - Speech recognition
      - Emotion detection
    """
    y, sr = librosa.load(audio_path)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Calculate statistics for each coefficient
    mfcc_stats = {}
    for i in range(n_mfcc):
        mfcc_stats[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        mfcc_stats[f'mfcc_{i}_std'] = np.std(mfccs[i])

    return mfcc_stats
```

---

### 4. **Temporal Characteristics - 時域特徵**

#### A. **Speaking Rate - 說話速度**
```python
def analyze_speaking_rate(audio_path):
    """
    Analyze how fast someone speaks

    Measured as:
      - Syllables per second
      - Pauses between words
    """
    y, sr = librosa.load(audio_path)

    # Detect speech/non-speech using energy
    energy = librosa.feature.rms(y=y)[0]
    threshold = np.mean(energy) * 0.3

    # Find speech segments
    is_speech = energy > threshold

    # Count transitions (rough syllable estimate)
    transitions = np.diff(is_speech.astype(int))
    syllable_count = np.sum(transitions == 1)

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    return {
        'syllables_per_second': syllable_count / duration,
        'speech_ratio': np.sum(is_speech) / len(is_speech),
        'avg_pause_duration': calculate_avg_pause(is_speech, sr)
    }
```

#### B. **Jitter and Shimmer - 音高/音量抖動**
- **Jitter**: Variation in fundamental frequency (pitch instability)
- **Shimmer**: Variation in amplitude (volume instability)
- Higher values = rougher, less smooth voice

```python
import parselmouth

def extract_jitter_shimmer(audio_path):
    """
    Extract voice quality measures

    Low jitter/shimmer = smooth, stable voice
    High jitter/shimmer = rough, unstable voice
    """
    sound = parselmouth.Sound(audio_path)

    # Create pitch object
    pitch = sound.to_pitch()

    # Point process for jitter calculation
    point_process = parselmouth.praat.call(
        sound, "To PointProcess (periodic, cc)", 75, 600
    )

    # Jitter (pitch variation)
    jitter_local = parselmouth.praat.call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )

    # Shimmer (amplitude variation)
    shimmer_local = parselmouth.praat.call(
        [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    return {
        'jitter': jitter_local,
        'shimmer': shimmer_local
    }
```

---

### 5. **Prosody - 韻律特徵**

**What is it?**
- Rhythm, stress, and intonation of speech
- How voice "flows"
- Emotional expressiveness

**Components**:
```
Intonation: Pitch pattern over time
  - Rising intonation: Questions (語調上揚)
  - Falling intonation: Statements (語調下降)

Stress: Emphasis on certain syllables
  - Louder, higher pitch, longer duration

Rhythm: Timing patterns
  - Fast bursts vs. slow deliberate speech
```

**Extraction**:
```python
def extract_prosody(audio_path):
    """
    Analyze prosodic features
    """
    y, sr = librosa.load(audio_path)

    # Extract pitch contour
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=80, fmax=600, sr=sr
    )

    # Calculate prosodic features
    f0_clean = f0[~np.isnan(f0)]

    return {
        # Pitch range (wide = expressive, narrow = monotone)
        'pitch_range': np.max(f0_clean) - np.min(f0_clean),

        # Pitch variance (variability)
        'pitch_variance': np.var(f0_clean),

        # Intonation slope (rising vs falling)
        'intonation_slope': np.polyfit(
            range(len(f0_clean)), f0_clean, 1
        )[0],

        # Dynamic range (volume variation)
        'dynamic_range_db': librosa.amplitude_to_db(
            np.max(np.abs(y))
        ) - librosa.amplitude_to_db(np.median(np.abs(y)))
    }
```

---

## Voice Embeddings - Modern AI Approach

### **What are Voice Embeddings?**

Instead of manually extracting features, modern AI creates a "fingerprint" of a voice:
- High-dimensional vector (e.g., 256-512 dimensions)
- Captures all voice characteristics automatically
- Similar voices have similar embeddings

### **Popular Models**:

#### 1. **Resemblyzer**
```python
from resemblyzer import VoiceEncoder, preprocess_wav

def get_voice_embedding(audio_path):
    """
    Extract voice embedding using Resemblyzer

    Output: 256-dimensional vector
    """
    encoder = VoiceEncoder()

    # Preprocess audio
    wav = preprocess_wav(audio_path)

    # Get embedding
    embedding = encoder.embed_utterance(wav)

    return embedding  # numpy array of shape (256,)
```

**Use Case**:
```python
# Compare two voices
def voice_similarity(voice1_path, voice2_path):
    """
    Calculate similarity between two voices

    Returns: 0-1 score (1 = identical)
    """
    encoder = VoiceEncoder()

    emb1 = encoder.embed_utterance(preprocess_wav(voice1_path))
    emb2 = encoder.embed_utterance(preprocess_wav(voice2_path))

    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    return similarity
```

#### 2. **Pyannote Embedding**
```python
from pyannote.audio import Inference

def get_pyannote_embedding(audio_path):
    """
    Extract speaker embedding using Pyannote

    More robust for speaker diarization
    """
    model = Inference(
        "pyannote/embedding",
        use_auth_token="YOUR_HF_TOKEN"
    )

    embedding = model(audio_path)

    return embedding
```

---

## How to Distinguish Different Characters

### Step-by-Step Process:

#### **Step 1: Extract Features from Known Samples**

```python
def create_character_profile(character_name, sample_clips):
    """
    Create voice profile for a character

    Args:
        character_name: "endou_mamoru"
        sample_clips: List of audio files with this character

    Returns:
        Character voice profile
    """
    encoder = VoiceEncoder()

    # Extract embeddings from all clips
    embeddings = []
    for clip in sample_clips:
        wav = preprocess_wav(clip)
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)

    # Average embedding = character's "voice fingerprint"
    avg_embedding = np.mean(embeddings, axis=0)

    # Also extract traditional features
    traditional_features = {
        'f0': [],
        'formants': [],
        'mfcc': [],
        'spectral': []
    }

    for clip in sample_clips:
        traditional_features['f0'].append(extract_f0(clip))
        traditional_features['formants'].append(extract_formants(clip))
        traditional_features['mfcc'].append(extract_mfcc(clip))
        traditional_features['spectral'].append(
            extract_spectral_features(clip)
        )

    # Create profile
    profile = {
        'character_name': character_name,
        'embedding': avg_embedding,
        'embedding_std': np.std(embeddings, axis=0),
        'features': {
            k: {
                'mean': np.mean(v, axis=0),
                'std': np.std(v, axis=0)
            }
            for k, v in traditional_features.items()
        },
        'num_samples': len(sample_clips)
    }

    return profile
```

#### **Step 2: Identify Unknown Speakers**

```python
def identify_speaker(unknown_clip, character_profiles):
    """
    Identify which character is speaking in unknown clip

    Args:
        unknown_clip: Audio file
        character_profiles: Dict of character profiles

    Returns:
        (character_name, confidence)
    """
    encoder = VoiceEncoder()

    # Extract embedding from unknown clip
    unknown_emb = encoder.embed_utterance(preprocess_wav(unknown_clip))

    # Compare with all known characters
    similarities = {}
    for char_name, profile in character_profiles.items():
        # Cosine similarity
        sim = np.dot(unknown_emb, profile['embedding']) / (
            np.linalg.norm(unknown_emb) * np.linalg.norm(profile['embedding'])
        )
        similarities[char_name] = sim

    # Best match
    best_match = max(similarities, key=similarities.get)
    confidence = similarities[best_match]

    return best_match, confidence
```

#### **Step 3: Clustering Unknown Speakers**

If you don't have labeled samples:
```python
from sklearn.cluster import AgglomerativeClustering

def cluster_speakers(audio_clips, n_speakers=None):
    """
    Automatically group clips by speaker

    Args:
        audio_clips: List of audio files
        n_speakers: Number of speakers (None = auto-detect)

    Returns:
        {speaker_id: [clips]}
    """
    encoder = VoiceEncoder()

    # Extract embeddings
    embeddings = []
    for clip in audio_clips:
        wav = preprocess_wav(clip)
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Cluster
    if n_speakers is None:
        # Auto-detect using silhouette score
        n_speakers = estimate_num_speakers(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=n_speakers,
        affinity='cosine',
        linkage='average'
    )

    labels = clustering.fit_predict(embeddings)

    # Group clips by speaker
    speakers = {}
    for i, label in enumerate(labels):
        if label not in speakers:
            speakers[label] = []
        speakers[label].append(audio_clips[i])

    return speakers
```

---

## Visualizing Voice Characteristics

### 1. **Spectrogram - 頻譜圖**
```python
import librosa.display
import matplotlib.pyplot as plt

def plot_spectrogram(audio_path, output_path):
    """
    Visualize frequency content over time

    X-axis: Time
    Y-axis: Frequency
    Color: Intensity (darker = louder)
    """
    y, sr = librosa.load(audio_path)

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        S_db, sr=sr, x_axis='time', y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
```

### 2. **Pitch Contour - 音高輪廓**
```python
def plot_pitch_contour(audio_path, output_path):
    """
    Visualize pitch changes over time

    Shows how voice goes up and down
    """
    y, sr = librosa.load(audio_path)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=80, fmax=600, sr=sr
    )

    times = librosa.times_like(f0, sr=sr)

    plt.figure(figsize=(12, 4))
    plt.plot(times, f0, label='f0', color='blue', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch Contour')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
```

### 3. **Formant Visualization**
```python
def plot_formants(audio_path, output_path):
    """
    Visualize formant frequencies

    Shows vocal tract resonances
    """
    sound = parselmouth.Sound(audio_path)
    formant = sound.to_formant_burg()

    # Get formant values over time
    times = formant.ts()
    f1_values = [formant.get_value_at_time(1, t) for t in times]
    f2_values = [formant.get_value_at_time(2, t) for t in times]
    f3_values = [formant.get_value_at_time(3, t) for t in times]

    plt.figure(figsize=(12, 6))
    plt.plot(times, f1_values, label='F1', alpha=0.7)
    plt.plot(times, f2_values, label='F2', alpha=0.7)
    plt.plot(times, f3_values, label='F3', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Formant Frequencies')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
```

### 4. **Voice Profile Comparison**
```python
def plot_voice_comparison(profiles, output_path):
    """
    Compare multiple character voice profiles

    Radar chart showing different characteristics
    """
    import matplotlib.pyplot as plt
    from math import pi

    # Features to compare
    features = ['Mean F0', 'F0 Range', 'Jitter', 'Shimmer',
                'Spectral Centroid', 'Speaking Rate']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Number of features
    N = len(features)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Plot for each character
    for char_name, profile in profiles.items():
        values = [
            profile['features']['f0']['mean'],
            profile['features']['f0']['std'],
            profile['features']['jitter'],
            profile['features']['shimmer'],
            profile['features']['spectral']['spectral_centroid_mean'],
            profile['features']['speaking_rate']
        ]

        # Normalize to 0-1
        values = [(v - min(values)) / (max(values) - min(values))
                  for v in values]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=char_name)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Voice Profile Comparison')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
```

---

## Practical Example: Inazuma Eleven Characters

### Expected Voice Characteristics:

#### **Endou Mamoru (円堂守)**
```yaml
Expected Profile:
  f0_mean: 220-260 Hz        # Mid-range male voice
  f0_std: 30-40 Hz           # Moderate variation (energetic)
  speaking_rate: Medium-Fast # Enthusiastic, quick responses
  pitch_range: Wide          # Expressive, emotional
  jitter: Low                # Clear, stable voice
  formants:
    F1: ~600 Hz
    F2: ~1400 Hz
  emotion: Energetic, determined
  speech_pattern: Direct, passionate
```

#### **Gouenji Shuuya (豪炎寺修也)**
```yaml
Expected Profile:
  f0_mean: 180-220 Hz        # Lower, cooler voice
  f0_std: 20-30 Hz           # Less variation (calm)
  speaking_rate: Slow-Medium # Deliberate, composed
  pitch_range: Moderate      # Controlled emotion
  jitter: Very Low           # Smooth, stable
  formants:
    F1: ~550 Hz
    F2: ~1300 Hz
  emotion: Cool, serious
  speech_pattern: Measured, thoughtful
```

#### **Kidou Yuuto (鬼道有人)**
```yaml
Expected Profile:
  f0_mean: 200-240 Hz        # Tactical, intelligent
  f0_std: 25-35 Hz           # Strategic variation
  speaking_rate: Medium      # Clear communication
  pitch_range: Moderate      # Balanced
  jitter: Low                # Clear diction
  formants:
    F1: ~580 Hz
    F2: ~1350 Hz
  emotion: Analytical, confident
  speech_pattern: Strategic, precise
```

---

## Required Python Libraries

```bash
# Core audio processing
pip install librosa soundfile scipy

# Voice features
pip install praat-parselmouth  # For formants, jitter, shimmer
pip install crepe              # For pitch extraction

# Voice embeddings
pip install resemblyzer         # Voice encoder
pip install pyannote.audio     # Speaker diarization

# Visualization
pip install matplotlib seaborn

# ML utilities
pip install scikit-learn numpy pandas
```

---

## Next Steps

I'll create:
1. **Complete voice analysis tool** that extracts all these features
2. **Visualization dashboard** to compare characters
3. **Automatic speaker clustering** for Inazuma Eleven episodes
4. **Voice profile database** for each character

Would you like me to implement the complete voice analyzer now?
