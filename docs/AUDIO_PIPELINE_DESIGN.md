# Audio Processing Pipeline Design
## Voice Cloning & Audio Generation for Anime Characters

---

## Overview

This pipeline enables extraction, processing, and generation of character voices from anime episodes:

1. **Extract** audio from video files
2. **Separate** vocals from background music/effects
3. **Identify** different speakers (characters)
4. **Clean** and filter voice samples
5. **Train** voice conversion models (RVC/SO-VITS)
6. **Generate** new speech with character voices

---

## Technology Stack

### Audio Extraction
- **FFmpeg**: Extract audio tracks from video
- **Pydub**: Audio manipulation and editing

### Voice Separation
- **Demucs v4**: SOTA audio source separation
  - HuggingFace: `facebook/htdemucs`
- **UVR5 (Ultimate Vocal Remover)**: Alternative/backup
  - Better for some anime audio

### Speaker Diarization
- **Pyannote-audio**: Speaker identification and segmentation
  - HuggingFace: `pyannote/speaker-diarization`
- **Resemblyzer**: Voice embedding for clustering

### Noise Reduction
- **Noisereduce**: Python noise reduction
- **RNNoise**: Deep learning denoise

### Voice Cloning/Conversion
- **RVC (Retrieval-based Voice Conversion)**: Primary choice
  - HuggingFace: `RVC-Project/Retrieval-based-Voice-Conversion-WebUI`
  - Pros: High quality, fast inference, less data needed

- **SO-VITS-SVC**: Alternative for singing voice
  - Better for songs/character themes

- **VITS**: Text-to-Speech option
  - For generating from text scripts

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO PROCESSING PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Episode Videos                                                  │
│      ↓                                                           │
│  [Audio Extraction] ─────────────→ episode_XX.wav                │
│      ↓                                                           │
│  [Voice Separation (Demucs)]                                     │
│      ↓                    ↓                                      │
│  Vocals.wav          BGM+SFX.wav                                 │
│      ↓                                                           │
│  [Speaker Diarization]                                           │
│      ↓                                                           │
│  Speaker 1 clips                                                 │
│  Speaker 2 clips                                                 │
│  Speaker 3 clips                                                 │
│      ↓                                                           │
│  [Manual Character Labeling]                                     │
│      ↓                                                           │
│  Character Voice Clips                                           │
│   - endou_mamoru/                                                │
│   - utsunomiya_toramaru/                                         │
│   - gouenji_shuuya/                                              │
│      ↓                                                           │
│  [Quality Filter & Noise Reduction]                              │
│      ↓                                                           │
│  Clean Voice Dataset (20-30 min per character)                   │
│      ↓                                                           │
│  [RVC Training]                                                  │
│      ↓                                                           │
│  Character Voice Model (.pth)                                    │
│      ↓                                                           │
│  [Voice Generation]                                              │
│   - Input: Reference voice + target text/audio                   │
│   - Output: Character saying new dialogue                        │
│      ↓                                                           │
│  Generated Character Speech                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Steps

### Step 1: Audio Extraction

**Tool**: FFmpeg

**Purpose**: Extract audio track from video files

**Implementation**:
```python
import subprocess
from pathlib import Path

def extract_audio(video_path: Path, output_path: Path,
                 sample_rate: int = 44100, channels: int = 1):
    """
    Extract audio from video file

    Args:
        video_path: Input video
        output_path: Output audio file (.wav)
        sample_rate: Audio sample rate (44100 or 48000)
        channels: 1 = mono, 2 = stereo
    """
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',                          # No video
        '-acodec', 'pcm_s16le',        # PCM 16-bit
        '-ar', str(sample_rate),        # Sample rate
        '-ac', str(channels),           # Channels
        '-y',                           # Overwrite
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
```

**Output**: `episode_001.wav` (~150MB per 20-min episode)

---

### Step 2: Voice Separation

**Tool**: Demucs v4 (htdemucs)

**Purpose**: Separate vocals from background music and sound effects

**Why Demucs?**
- State-of-the-art quality
- 4-stem separation: vocals, drums, bass, other
- Fast with GPU
- Good for anime audio

**Implementation**:
```python
import demucs.separate

def separate_vocals(audio_path: Path, output_dir: Path):
    """
    Separate vocals from BGM/SFX using Demucs

    Output structure:
        output_dir/
            htdemucs/
                episode_001/
                    vocals.wav      # Character voices
                    drums.wav       # Percussion
                    bass.wav        # Bass line
                    other.wav       # BGM, SFX
    """
    # Use htdemucs model (best quality)
    demucs.separate.main([
        '--two-stems', 'vocals',  # Only vocals vs accompaniment
        '-n', 'htdemucs',         # Model name
        '--out', str(output_dir),
        str(audio_path)
    ])
```

**HuggingFace Model**: `facebook/htdemucs`

**Download**:
```bash
# Demucs downloads models automatically on first use
# Or manually:
python -m demucs.separate --help  # Will trigger download
```

**Output**:
- `vocals.wav` - Character dialogue
- `other.wav` - BGM and sound effects (can be reused!)

---

### Step 3: Speaker Diarization

**Tool**: Pyannote-audio

**Purpose**: Identify different speakers (characters) in the vocal track

**How it works**:
1. Detect speech segments
2. Extract voice embeddings
3. Cluster similar voices
4. Assign speaker IDs

**Implementation**:
```python
from pyannote.audio import Pipeline

def identify_speakers(vocal_audio: Path, output_dir: Path,
                     num_speakers: int = None):
    """
    Identify and separate different speakers

    Args:
        vocal_audio: vocals.wav from separation
        output_dir: Where to save individual speaker clips
        num_speakers: Expected number (None = auto-detect)

    Returns:
        Dictionary mapping speaker IDs to audio clips
    """
    # Load pretrained pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="YOUR_HF_TOKEN"  # Need HF token
    )

    # Run diarization
    diarization = pipeline(vocal_audio, num_speakers=num_speakers)

    # Extract clips for each speaker
    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # turn.start, turn.end = timestamp
        clip = extract_segment(vocal_audio, turn.start, turn.end)

        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(clip)

    return speakers
```

**HuggingFace Model**: `pyannote/speaker-diarization`

**Requirements**:
```bash
pip install pyannote.audio
# Need HuggingFace token (free): https://huggingface.co/settings/tokens
```

**Output**:
```
output_dir/
    speaker_0/  # Need manual labeling: "Who is this?"
        clip_001.wav
        clip_002.wav
        ...
    speaker_1/
        clip_001.wav
        ...
    speaker_2/
        ...
```

---

### Step 4: Character Labeling (Manual)

**Challenge**: Diarization gives us "speaker_0", "speaker_1", etc.
We need to identify which speaker is which character.

**Solution**: Manual labeling tool

```python
def label_speakers(speaker_clips_dir: Path):
    """
    Interactive tool to listen and label speakers

    Workflow:
        1. Play clip from speaker_0
        2. User identifies: "This is Endou Mamoru"
        3. Rename/move clips to characters/endou_mamoru/
        4. Repeat for all speakers
    """
    # Implementation would use:
    # - Audio playback (sounddevice)
    # - CLI or GUI interface
    # - Automatic renaming/organizing
    pass
```

**Better Approach**: Character voice reference

If you have known samples of each character:
```python
from resemblyzer import VoiceEncoder, preprocess_wav

def match_speakers_to_characters(speaker_clips: dict,
                                 character_references: dict):
    """
    Automatically match speakers to characters using voice similarity

    Args:
        speaker_clips: {speaker_id: [clips]}
        character_references: {character_name: reference_wav}

    Returns:
        {character_name: [clips]}
    """
    encoder = VoiceEncoder()

    # Encode character references
    char_embeddings = {}
    for char_name, ref_wav in character_references.items():
        wav = preprocess_wav(ref_wav)
        char_embeddings[char_name] = encoder.embed_utterance(wav)

    # Match speakers
    result = {}
    for speaker_id, clips in speaker_clips.items():
        # Average embedding for this speaker
        speaker_embedding = get_average_embedding(clips, encoder)

        # Find closest character
        best_match = find_closest(speaker_embedding, char_embeddings)
        result[best_match] = clips

    return result
```

---

### Step 5: Quality Filtering & Noise Reduction

**Purpose**: Clean up voice clips for training

**Filters**:
1. **Duration filter**: Remove clips < 1s or > 15s
2. **Silence removal**: Trim leading/trailing silence
3. **SNR filter**: Remove low signal-to-noise ratio clips
4. **Volume normalization**: Consistent loudness
5. **Noise reduction**: Remove background noise

**Implementation**:
```python
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

def clean_audio_clip(audio_path: Path, output_path: Path):
    """
    Apply all cleaning steps to an audio clip
    """
    # Load audio
    audio = AudioSegment.from_wav(audio_path)

    # 1. Trim silence
    trim_ms = detect_leading_silence(audio, silence_threshold=-40)
    audio = audio[trim_ms:]

    # Trim from end
    audio_reversed = audio.reverse()
    trim_ms = detect_leading_silence(audio_reversed, silence_threshold=-40)
    audio = audio[:-trim_ms] if trim_ms > 0 else audio

    # 2. Check duration
    duration_s = len(audio) / 1000.0
    if duration_s < 1.0 or duration_s > 15.0:
        return False  # Skip this clip

    # 3. Normalize volume
    audio = audio.normalize()

    # 4. Noise reduction
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    reduced = nr.reduce_noise(y=samples, sr=sample_rate)

    # 5. Export
    cleaned_audio = AudioSegment(
        reduced.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio.sample_width,
        channels=1
    )
    cleaned_audio.export(output_path, format="wav")

    return True
```

**Quality Targets**:
- **Total duration per character**: 20-30 minutes
- **Clip length**: 1-15 seconds
- **Sample rate**: 44100 Hz
- **Format**: Mono WAV, 16-bit PCM

---

### Step 6: RVC Training

**Tool**: RVC (Retrieval-based Voice Conversion)

**Why RVC?**
- High quality results
- Needs less data (20-30 min vs hours)
- Fast inference
- Good for anime voices

**Training Process**:

```python
# Using RVC-WebUI or rvc-python

def train_rvc_model(
    character_name: str,
    voice_clips_dir: Path,
    output_model_path: Path,
    epochs: int = 500
):
    """
    Train RVC model for character voice

    Args:
        character_name: e.g., "endou_mamoru"
        voice_clips_dir: Clean voice samples
        output_model_path: Where to save .pth model
        epochs: Training iterations (300-500 typical)
    """
    # RVC training steps:
    # 1. Preprocess audio
    # 2. Extract features (pitch, formant, etc.)
    # 3. Train model
    # 4. Generate index for retrieval

    # This would use RVC's training script
    # or Python API if available
    pass
```

**RVC Models on HuggingFace**:
- Base models: `RVC-Project/Retrieval-based-Voice-Conversion-WebUI`
- Pre-trained checkpoints available

**Training Requirements**:
- **GPU**: Recommended (training is slow on CPU)
- **VRAM**: 6GB+ for training
- **Time**: 2-4 hours per character (500 epochs)
- **Data**: 20-30 minutes of clean speech

**Training Parameters**:
```yaml
sample_rate: 40000  # RVC uses 40k or 48k
f0_method: "crepe"  # Pitch extraction (most accurate)
hop_length: 128
batch_size: 8
save_every_epoch: 50
total_epoch: 500
```

---

### Step 7: Voice Generation

**Use Cases**:

#### A. Voice Conversion (Audio → Audio)
```python
def convert_voice(source_audio: Path,
                 character_model: Path,
                 output_path: Path,
                 pitch_shift: int = 0):
    """
    Convert any voice to character's voice

    Args:
        source_audio: Input speech (any voice)
        character_model: Trained RVC model (.pth)
        pitch_shift: Semitones to shift pitch

    Example:
        # Make your voice sound like Endou
        convert_voice(
            "my_recording.wav",
            "models/endou_mamoru.pth",
            "endou_saying_it.wav"
        )
    """
    # Use RVC inference
    pass
```

#### B. Text-to-Speech (Text → Audio)
```python
def text_to_character_speech(text: str,
                            character_model: Path,
                            output_path: Path):
    """
    Generate character speech from text

    Workflow:
        1. Generate base speech with TTS (e.g., VITS)
        2. Convert to character voice with RVC

    Args:
        text: What character should say
        character_model: RVC model

    Example:
        text_to_character_speech(
            "Let's play soccer!",
            "models/endou_mamoru.pth",
            "endou_soccer.wav"
        )
    """
    # Step 1: TTS
    base_speech = generate_tts(text)

    # Step 2: Voice conversion
    convert_voice(base_speech, character_model, output_path)
```

---

## Implementation Plan for Inazuma Eleven

### Phase 1: Audio Extraction (Week 1)

**Tasks**:
1. Create audio extraction tool
2. Extract audio from all 126 episodes
3. Organize by episode

**Script**:
```bash
python shared/tools/extraction/audio_extractor.py \
  --series inazuma-eleven \
  --input-dir warehouse/raw_data/inazuma-eleven/raw_videos \
  --output-dir warehouse/processed/japanese-anime/inazuma-eleven/audio
```

**Output**: 126 WAV files (~20 GB)

---

### Phase 2: Voice Separation (Week 1-2)

**Tasks**:
1. Setup Demucs
2. Separate vocals from all episodes
3. Archive BGM tracks (reusable!)

**Script**:
```bash
python shared/tools/processing/audio/voice_separator.py \
  --series inazuma-eleven \
  --model htdemucs \
  --workers 4
```

**Output**:
- Vocals: 126 files (~10 GB)
- BGM: 126 files (~10 GB) - **Save these for future use!**

---

### Phase 3: Speaker Identification (Week 2)

**Tasks**:
1. Run diarization on 5-10 sample episodes
2. Identify main characters' voices
3. Create reference voice samples
4. Auto-match speakers across all episodes

**Target Characters**:
- Endou Mamoru (protagonist, most dialogue)
- Gouenji Shuuya
- Kidou Yuuto
- Kazemaru Ichirouta
- Fubuki Shirou

**Manual Work**: ~2-4 hours to label initial samples

---

### Phase 4: Dataset Preparation (Week 3)

**Tasks**:
1. Extract clips for each character
2. Quality filtering
3. Noise reduction
4. Organize training datasets

**Target**: 20-30 min per character

---

### Phase 5: RVC Training (Week 3-4)

**Tasks**:
1. Train RVC models for each character
2. Test voice quality
3. Fine-tune if needed

**Resources**:
- GPU: Training can run overnight
- Can train multiple characters in parallel if VRAM allows

---

### Phase 6: Voice Generation Tool (Week 4)

**Tasks**:
1. Create voice generation interface
2. Integrate with existing tools
3. Test end-to-end workflow

**Demo**:
```bash
# Generate Endou saying custom dialogue
python shared/tools/generation/audio_generator.py \
  --series inazuma-eleven \
  --character endou_mamoru \
  --text "必殺技、ゴッドハンド！" \
  --output endou_godhand.wav
```

---

## HuggingFace Models Required

### Voice Separation
```yaml
demucs:
  model_id: facebook/htdemucs
  size: 2.4 GB
  download: automatic on first use
```

### Speaker Diarization
```yaml
pyannote_diarization:
  model_id: pyannote/speaker-diarization
  size: 300 MB
  requires: HuggingFace token (free)

pyannote_embedding:
  model_id: pyannote/embedding
  size: 150 MB
```

### RVC Base Models
```yaml
rvc_base:
  model_id: RVC-Project/Retrieval-based-Voice-Conversion-WebUI
  size: varies
  note: Multiple pretrained models available
```

### TTS (Optional for text-to-speech)
```yaml
vits_japanese:
  model_id: rinna/japanese-gpt-1b-vits
  size: 4 GB
  purpose: Japanese TTS for base speech generation
```

---

## Expected Results

### After Completion:

1. **Character Voice Models**:
   - `endou_mamoru.pth` (RVC model)
   - `gouenji_shuuya.pth`
   - `kidou_yuuto.pth`
   - etc.

2. **Capabilities**:
   - Clone any character's voice
   - Generate new dialogue
   - Voice-over for animations
   - Dubbing in different languages (convert to character voice)

3. **Reusable Assets**:
   - BGM tracks (for video generation)
   - Sound effects library
   - Character voice datasets

---

## Next Steps

Want me to:
1. **Start implementing audio extraction tool** ✅ (Ready to code)
2. **Setup Demucs integration** ✅ (Ready to code)
3. **Create speaker diarization pipeline** ✅ (Ready to code)
4. **Test on sample Inazuma Eleven episode** (Quick validation)

Which would you like me to start with?
