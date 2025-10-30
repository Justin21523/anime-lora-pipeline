# Audio Processing Pipeline - Complete Usage Guide

This guide provides step-by-step instructions for processing anime audio for voice cloning and character voice analysis.

## Table of Contents

1. [Installation](#installation)
2. [Pipeline Overview](#pipeline-overview)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [Tool Reference](#tool-reference)
5. [Batch Processing](#batch-processing)
6. [Understanding Voice Analysis Results](#understanding-results)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Required Packages

```bash
# Activate your conda environment
conda activate blip2-env

# Core audio processing
pip install librosa soundfile audioread

# Voice separation (Demucs)
pip install demucs

# Voice analysis
pip install matplotlib numpy scipy

# Optional: Formant analysis (highly recommended)
pip install praat-parselmouth

# Optional: Voice embeddings for speaker identification
pip install resemblyzer pyannote-audio
```

### Verify Installation

```bash
# Check FFmpeg (required for audio extraction)
ffmpeg -version

# Check Demucs
demucs --help

# Test Python imports
python3 -c "import librosa; import demucs; print('✓ Audio libraries ready')"
```

---

## Pipeline Overview

The complete audio processing pipeline consists of 3 main stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                     AUDIO PROCESSING PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. EXTRACTION                                                   │
│     Video Files (MP4/MKV) → Audio Files (WAV/FLAC)              │
│     Tool: audio_extractor.py                                     │
│                                                                  │
│  2. SEPARATION                                                   │
│     Mixed Audio → Vocals Only (BGM removed)                      │
│     Tool: voice_separator.py (uses Demucs)                       │
│                                                                  │
│  3. ANALYSIS                                                     │
│     Vocals → Voice Characteristics (pitch, formants, etc.)       │
│     Tool: voice_analyzer.py                                      │
│                                                                  │
│  4. IDENTIFICATION (Optional)                                    │
│     Group voices by character (speaker diarization)              │
│     Tool: speaker_diarization.py (future)                        │
│                                                                  │
│  5. TRAINING (Future)                                            │
│     Character voices → Voice cloning models (RVC)                │
│     Tool: rvc_trainer.py (future)                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Workflow

### Example: Processing Inazuma Eleven Episodes

We'll process episodes to extract and analyze character voices.

#### Step 1: Extract Audio from Videos

Extract audio from all episode videos to WAV format.

```bash
# Navigate to project directory
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora

# Run audio extraction
conda run -n blip2-env python3 scripts/tools/audio_extractor.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/raw_videos \
  --output-dir /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/audio \
  --format wav \
  --sample-rate 44100 \
  --channels 1 \
  --workers 16

# Expected output:
# - 126 WAV files (one per episode)
# - ~8 GB total size
# - Processing time: ~20-30 minutes
```

**Parameters Explained:**
- `--format wav`: Uncompressed format, best for processing
- `--sample-rate 44100`: Standard audio quality (44.1kHz)
- `--channels 1`: Mono audio (sufficient for voice, saves space)
- `--workers 16`: Use all 16 CPU cores for parallel processing

**Output Structure:**
```
ai_warehouse/cache/inazuma-eleven/audio/
├── episode_001.wav
├── episode_002.wav
├── episode_003.wav
...
└── episode_126.wav
```

#### Step 2: Separate Vocals from Background

Remove background music and sound effects, keeping only voices.

```bash
# Run voice separation
conda run -n blip2-env python3 scripts/tools/voice_separator.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/audio \
  --output-dir /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/separated \
  --model htdemucs \
  --shifts 1 \
  --device cuda

# Expected output:
# - 126 vocals-only WAV files
# - ~4 GB total size
# - Processing time: ~2-3 hours (GPU)
```

**Parameters Explained:**
- `--model htdemucs`: Best quality model (recommended)
- `--shifts 1`: Quality setting (1=fast, 5=best quality)
- `--device cuda`: Use GPU acceleration (much faster)
- Default `--two-stems`: Extract vocals only (faster than full separation)

**Output Structure:**
```
ai_warehouse/cache/inazuma-eleven/separated/
├── htdemucs/
│   ├── episode_001/
│   │   ├── vocals.wav      ← Voice only
│   │   └── no_vocals.wav   ← BGM only
│   ├── episode_002/
│   ...
└── vocals/                  ← Organized flat directory
    ├── episode_001_vocals.wav
    ├── episode_002_vocals.wav
    ...
```

**Quality Check:**
```bash
# Listen to a sample to verify separation quality
ffplay ai_warehouse/cache/inazuma-eleven/separated/vocals/episode_001_vocals.wav

# Compare with original
ffplay ai_warehouse/cache/inazuma-eleven/audio/episode_001.wav
```

#### Step 3: Analyze Voice Characteristics

Extract detailed voice features to understand character voices.

```bash
# Run voice analysis
conda run -n blip2-env python3 scripts/tools/voice_analyzer.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/separated/vocals \
  --output-csv /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/voice_analysis.csv \
  --formants \
  --visualize \
  --viz-dir /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/visualizations

# Expected output:
# - voice_analysis.csv with all voice features
# - 126 visualization images
# - Processing time: ~30-60 minutes
```

**Parameters Explained:**
- `--output-csv`: Save all features to CSV for analysis
- `--formants`: Extract formant frequencies (requires praat-parselmouth)
- `--visualize`: Create visual plots for each episode
- `--viz-dir`: Directory to save visualizations

**Output Files:**

1. **CSV File** (`voice_analysis.csv`):
```csv
filename,duration,f0_mean,f0_std,f0_min,f0_max,f0_range,f1_mean,f2_mean,f3_mean,...
episode_001_vocals.wav,1380.5,220.3,45.2,120.5,380.2,259.7,650.2,1720.5,2850.3,...
episode_002_vocals.wav,1420.8,215.8,42.8,125.3,375.8,250.5,655.8,1715.2,2845.8,...
...
```

2. **Visualizations** (PNG files):
   - Waveform plot
   - Spectrogram (frequency over time)
   - Pitch contour (F0 over time)
   - MFCCs (voice fingerprint)

**Understanding the Results:**

The analysis will print a summary like this:
```
Voice Comparison Summary
================================================================================

Filename                                 Pitch (Hz)      Range      Brightness
--------------------------------------------------------------------------------
episode_015_vocals.wav                   180.5 Hz        145.2      2.85 kHz
episode_023_vocals.wav                   185.3 Hz        152.8      2.92 kHz
episode_042_vocals.wav                   218.7 Hz        201.5      3.15 kHz
episode_051_vocals.wav                   245.2 Hz        185.3      3.28 kHz

Interpretation:
  - Lower pitch (< 180 Hz): Typically male voices
  - Higher pitch (> 200 Hz): Typically female/child voices
  - Larger range: More expressive/emotional delivery
  - Higher brightness: Sharper, crisper voice quality
```

---

## Tool Reference

### 1. audio_extractor.py

**Purpose:** Extract audio from video files

**Basic Usage:**
```bash
python audio_extractor.py <video_directory> [OPTIONS]
```

**Common Options:**
```bash
--output-dir PATH       # Output directory (default: video_dir/../audio)
--format FORMAT         # Audio format: wav, flac, mp3 (default: wav)
--sample-rate RATE      # Sample rate in Hz (default: 44100)
--channels NUM          # 1=mono, 2=stereo (default: 1)
--workers NUM           # Parallel workers (default: CPU_count/2)
--episode-pattern REGEX # Episode number extraction pattern
--no-normalize          # Disable audio normalization
```

**Examples:**

```bash
# Basic extraction
python audio_extractor.py /path/to/videos

# High-quality FLAC with custom sample rate
python audio_extractor.py /path/to/videos --format flac --sample-rate 48000

# For Yokai Watch (S1.01 format)
python audio_extractor.py /path/to/videos --episode-pattern "S\d+\.(\d+)"

# Fast extraction with 16 workers
python audio_extractor.py /path/to/videos --workers 16
```

### 2. voice_separator.py

**Purpose:** Separate vocals from background music/SFX using Demucs

**Basic Usage:**
```bash
python voice_separator.py <audio_directory> [OPTIONS]
```

**Common Options:**
```bash
--output-dir PATH       # Output directory (default: audio_dir/../separated)
--model MODEL           # Demucs model: htdemucs, htdemucs_ft, mdx_extra
--shifts NUM            # Quality: 1=fast, 5=best (default: 1)
--device DEVICE         # cuda or cpu (default: cuda)
--full-stems            # Extract all stems (vocals, bass, drums, other)
--jobs NUM              # CPU threads for processing
```

**Examples:**

```bash
# Basic separation (vocals only, GPU)
python voice_separator.py /path/to/audio

# High quality with fine-tuned model
python voice_separator.py /path/to/audio --model htdemucs_ft --shifts 5

# CPU processing with multiple threads
python voice_separator.py /path/to/audio --device cpu --jobs 8

# Extract all instrument tracks
python voice_separator.py /path/to/audio --full-stems
```

**Model Comparison:**
| Model | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| htdemucs | Excellent | Moderate | Best for voice cloning (recommended) |
| htdemucs_ft | Best | Slow | When quality is critical |
| mdx_extra | Good | Fast | Quick testing |

### 3. voice_analyzer.py

**Purpose:** Extract and analyze voice characteristics

**Basic Usage:**
```bash
python voice_analyzer.py <vocals_directory> [OPTIONS]
```

**Common Options:**
```bash
--output-csv PATH       # Save results to CSV file
--formants              # Extract formant frequencies (requires praat-parselmouth)
--visualize             # Create visualization plots
--viz-dir PATH          # Directory for visualizations
```

**Examples:**

```bash
# Basic analysis
python voice_analyzer.py /path/to/vocals

# Full analysis with all features
python voice_analyzer.py /path/to/vocals --formants --output-csv results.csv

# Create visualizations
python voice_analyzer.py /path/to/vocals --visualize --viz-dir ./plots
```

**Extracted Features:**

| Category | Features | Purpose |
|----------|----------|---------|
| **Pitch** | F0 (mean, std, min, max, range) | Voice pitch, gender identification |
| **Formants** | F1, F2, F3 | Vowel quality, voice timbre |
| **Spectral** | Centroid, bandwidth, rolloff, ZCR | Voice brightness, roughness |
| **MFCCs** | 13 coefficients | Voice fingerprint |
| **Temporal** | Speaking rate, pause ratio | Speech rhythm |
| **Energy** | RMS (mean, std) | Loudness, vocal effort |

---

## Batch Processing

### Automated Pipeline Script

Create a batch script to run the entire pipeline automatically:

**File:** `scripts/batch/inazuma_eleven_audio.sh`

```bash
#!/bin/bash
# Complete audio processing pipeline for Inazuma Eleven

SERIES="inazuma-eleven"
BASE_DIR="/mnt/c/AI_LLM_projects/ai_warehouse"
VIDEO_DIR="$BASE_DIR/training_data/$SERIES/raw_videos"
CACHE_DIR="$BASE_DIR/cache/$SERIES"

echo "=========================================="
echo "Inazuma Eleven Audio Processing Pipeline"
echo "=========================================="
echo ""

# Step 1: Extract audio
echo "[1/3] Extracting audio from videos..."
conda run -n blip2-env python3 scripts/tools/audio_extractor.py \
  "$VIDEO_DIR" \
  --output-dir "$CACHE_DIR/audio" \
  --format wav \
  --sample-rate 44100 \
  --channels 1 \
  --workers 16

echo ""
echo "[1/3] ✓ Audio extraction complete"
echo ""

# Step 2: Separate vocals
echo "[2/3] Separating vocals from background..."
conda run -n blip2-env python3 scripts/tools/voice_separator.py \
  "$CACHE_DIR/audio" \
  --output-dir "$CACHE_DIR/separated" \
  --model htdemucs \
  --shifts 1 \
  --device cuda

echo ""
echo "[2/3] ✓ Voice separation complete"
echo ""

# Step 3: Analyze voices
echo "[3/3] Analyzing voice characteristics..."
conda run -n blip2-env python3 scripts/tools/voice_analyzer.py \
  "$CACHE_DIR/separated/vocals" \
  --output-csv "$CACHE_DIR/voice_analysis.csv" \
  --formants \
  --visualize \
  --viz-dir "$CACHE_DIR/visualizations"

echo ""
echo "[3/3] ✓ Voice analysis complete"
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Audio: $CACHE_DIR/audio"
echo "  Vocals: $CACHE_DIR/separated/vocals"
echo "  Analysis: $CACHE_DIR/voice_analysis.csv"
echo "  Visualizations: $CACHE_DIR/visualizations"
echo ""
```

**Usage:**
```bash
chmod +x scripts/batch/inazuma_eleven_audio.sh
bash scripts/batch/inazuma_eleven_audio.sh
```

---

## Understanding Results

### Interpreting Voice Features

#### 1. Fundamental Frequency (F0) - Pitch

**What it tells you:** The perceived pitch of the voice

**Typical Ranges:**
- Adult male: 85-180 Hz
- Adult female: 165-255 Hz
- Child: 250-300 Hz
- Anime characters (male): 150-220 Hz
- Anime characters (female): 200-280 Hz

**How to use it:**
- Group episodes by similar pitch to identify consistent characters
- High pitch variance = emotional/expressive scenes
- Low variance = calm/serious dialogue

**Example Interpretation:**
```
Episode 015: F0 = 185 Hz, range = 120 Hz  → Likely male character, calm speech
Episode 042: F0 = 245 Hz, range = 200 Hz  → Likely female/child, emotional scene
```

#### 2. Formants (F1, F2, F3)

**What they tell you:** Vocal tract shape, unique to each speaker

**Typical Values:**
- F1: 300-1000 Hz (jaw opening, vowel height)
- F2: 800-2500 Hz (tongue position, vowel frontness)
- F3: 2000-3500 Hz (voice quality, speaker identity)

**How to use it:**
- Formants are like a "voice fingerprint"
- Similar F1/F2/F3 = same character
- Different formants = different characters

**Example:**
```
Endou Mamoru (goalkeeper):
  F1 = 650 Hz, F2 = 1720 Hz, F3 = 2850 Hz

Gouenji Shuuya (striker):
  F1 = 620 Hz, F2 = 1680 Hz, F3 = 2800 Hz

Akihiko (child character):
  F1 = 720 Hz, F2 = 1850 Hz, F3 = 2950 Hz
```

#### 3. Spectral Centroid - Brightness

**What it tells you:** The "brightness" or "sharpness" of the voice

**Typical Values:**
- Low (< 2.5 kHz): Warm, mellow voice
- Medium (2.5-3.5 kHz): Normal voice
- High (> 3.5 kHz): Bright, sharp voice

**How to use it:**
- Identifies voice quality/timbre
- Energetic characters often have higher spectral centroid

#### 4. Speaking Rate

**What it tells you:** How fast the character speaks

**Typical Values:**
- Slow: < 3 syllables/second
- Normal: 3-5 syllables/second
- Fast: > 5 syllables/second

**How to use it:**
- Identify character personality traits
- Fast-talking characters vs calm characters

### Visualizations Guide

Each visualization shows different aspects of the voice:

#### 1. Waveform
```
Shows: Amplitude over time
Look for: Overall volume, pauses, speech patterns
```

#### 2. Spectrogram
```
Shows: Frequency content over time (colorful heatmap)
Look for: Voice harmonics, background noise, pitch patterns
Dark horizontal lines = formants (voice characteristics)
```

#### 3. Pitch Contour
```
Shows: F0 (pitch) over time
Look for:
  - Flat lines = monotone speech
  - Rising/falling = intonation patterns
  - High variance = emotional/expressive
```

#### 4. MFCCs
```
Shows: Compact representation of voice spectrum
Look for: Overall patterns (similar MFCCs = similar voices)
```

---

## Troubleshooting

### Problem: "demucs: command not found"

**Solution:**
```bash
conda activate blip2-env
pip install demucs
demucs --help  # Verify installation
```

### Problem: GPU out of memory during voice separation

**Solutions:**

1. Use CPU instead:
```bash
python voice_separator.py /path/to/audio --device cpu --jobs 8
```

2. Process files one at a time:
```bash
python voice_separator.py /path/to/audio --sequential
```

3. Use lighter model:
```bash
python voice_separator.py /path/to/audio --model mdx_extra
```

### Problem: Formant extraction fails

**Solution:**
```bash
# Install praat-parselmouth
pip install praat-parselmouth

# Or skip formants:
python voice_analyzer.py /path/to/vocals  # Don't use --formants flag
```

### Problem: Poor vocal separation quality

**Possible causes and solutions:**

1. **Too much background music:**
   - Use `--shifts 5` for better quality (slower)
   - Try `--model htdemucs_ft` (best model)

2. **Multiple speakers talking:**
   - This is expected; separation extracts all voices
   - Use speaker diarization to separate individual speakers

3. **Sound effects mixed with voice:**
   - Some SFX will remain (shouting, impacts)
   - This is normal and OK for training

### Problem: Visualizations not saving

**Solution:**
```bash
# Ensure matplotlib backend is correct
export MPLBACKEND=Agg

# Or install display backend
sudo apt-get install python3-tk
```

---

## Next Steps

After completing voice analysis:

1. **Review Results:**
   - Check CSV file for voice characteristics
   - View visualizations to understand voice patterns
   - Group episodes with similar voices

2. **Character Identification:**
   - Manually label audio segments by character
   - Or use speaker diarization (future tool)

3. **Voice Cloning:**
   - Select high-quality voice samples per character
   - Train RVC models (future implementation)
   - Generate new voice samples

4. **Dataset Curation:**
   - Filter out low-quality segments (background noise, crowd scenes)
   - Select clean dialogue samples
   - Organize by character for training

---

## References

- **Demucs:** https://github.com/facebookresearch/demucs
- **Librosa:** https://librosa.org/
- **Praat-parselmouth:** https://parselmouth.readthedocs.io/
- **RVC (Voice Cloning):** https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

---

**Last Updated:** 2025-10-25
**Version:** 1.0
