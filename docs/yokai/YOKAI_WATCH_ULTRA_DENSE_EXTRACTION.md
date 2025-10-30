# Yokai Watch Ultra-Dense Frame Extraction

## Problem Analysis

### Previous Extraction Issues

**Issue #1: Incomplete Coverage**
```
Files in dataset:    218 total (214 episodes + 4 movies)
Files extracted:     65 (only Season 1)
Missing:             153 files (71% of dataset!)

Season breakdown:
  ✓ S1: 65 episodes  → Extracted
  ✗ S2: 52 episodes  → Missing
  ✗ S3: 52 episodes  → Missing
  ✗ S4: 45 episodes  → Missing
  ✗ Movies: 4 films  → Missing
```

**Issue #2: Insufficient Frame Density**
```
Previous extraction captured ~2,400 frames/episode
Problem: Yokai summoning animations have rapid, subtle changes
Example: A 3-second summoning sequence at 30fps = 90 frames
  - Previous: Captured ~12 frames (13% coverage)
  - Result: Missing critical transformation steps
```

---

## Optimization Strategy

### Parameter Comparison

| Parameter | Previous | Ultra-Dense | Improvement |
|-----------|----------|-------------|-------------|
| **Scene Threshold** | 22 | **15** | +47% sensitivity |
| **Frames per Scene** | 5 | **8** | +60% coverage |
| **Interval** | 4 seconds | **2 seconds** | +100% temporal density |
| **Workers** | 8 | **16** | +100% speed |
| **Files Coverage** | 65 (30%) | **218 (100%)** | Complete dataset |

### Expected Output Comparison

| Metric | Previous | Ultra-Dense | Improvement |
|--------|----------|-------------|-------------|
| **Frames/Episode** | ~2,400 | ~4,500 | +87% |
| **Total Frames** | ~156,000 | ~980,000 | +529% |
| **Storage** | ~26 GB | ~170 GB | More data |
| **Processing Time** | 4 hours (65 ep) | 12-15 hours (218 files) | 3.5x more files |

---

## Why Ultra-Dense Extraction?

### 1. Yokai Summoning Sequences

Yokai Watch summoning animations are **extremely fast and detailed**:

```
Typical summoning sequence (3 seconds @ 30fps):
├─ Frame 0-15:   Watch opens, light emerges
├─ Frame 16-30:  Yokai medal spins, energy builds
├─ Frame 31-60:  Transformation sequence (rapid changes)
├─ Frame 61-75:  Yokai materializes
└─ Frame 76-90:  Final pose and effects

With previous extraction (4s interval):
  → Captured maybe 1-2 frames from this entire sequence ❌

With ultra-dense extraction (2s interval + scene detection):
  → Captures 15-20 frames, preserving the full animation ✅
```

### 2. Micro-Expression Changes

**Problem:** Character facial expressions change rapidly during emotional scenes.

**Example:** Jibanyan's reaction sequence (0.5 seconds)
```
Frame 1:  😐 Neutral
Frame 3:  😮 Surprised (eyes widen)
Frame 6:  😠 Angry (eyebrows furrow)
Frame 9:  😤 Determined (mouth opens)
Frame 12: 🔥 Battle ready (full expression)
```

**Previous:** Might capture frame 1 and 12 → Missing the transition
**Ultra-Dense:** Captures all intermediate frames → Full emotional arc

### 3. Attack Animation Details

**Problem:** Special attack animations are multi-phase and detailed.

**Example:** "Paws of Fury" attack animation
```
Phase 1 (0.5s): Wind-up pose
Phase 2 (0.3s): Energy gathering (particles appear)
Phase 3 (0.7s): Strike motion (multiple poses)
Phase 4 (0.5s): Impact effects
Phase 5 (0.5s): Recovery pose
```

Each phase has unique characteristics that are important for training:
- **Character LoRA:** Needs different poses from the same character
- **Effect models:** Needs energy/impact particle patterns
- **Motion models:** Needs frame-by-frame progression

---

## Scene Threshold Deep Dive

### What is Scene Threshold?

Scene threshold determines how sensitive the detector is to visual changes:

```
Threshold = Measure of pixel difference between consecutive frames

Lower threshold = More sensitive = More scenes detected
Higher threshold = Less sensitive = Fewer scenes detected
```

### Threshold Comparison

#### Previous: Threshold = 22

```
Scene detected when: ~22% of pixels change significantly

Example: Jibanyan summoning
├─ [Scene 1] Normal background
│   Frame 1-2-3-4-5 (no scene change - below threshold)
├─ [SCENE CHANGE] Watch opens (23% pixel change) ← Detected
│   Frame 6-7-8-9-10
├─ [Missed!] Medal spins (18% change) ← Too subtle, not detected
│   Frame 11-12-13-14-15
├─ [SCENE CHANGE] Transformation flash (35% change) ← Detected
│   Frame 16-17-18-19-20
└─ [Missed!] Materialization (20% change) ← Too subtle, not detected
    Frame 21-22-23-24-25

Result: Captured 2 scene changes, missed 2 critical moments
```

#### Ultra-Dense: Threshold = 15

```
Scene detected when: ~15% of pixels change significantly

Example: Same Jibanyan summoning
├─ [Scene 1] Normal background
│   Frame 1-2-3-4-5
├─ [SCENE CHANGE] Watch opens (23% change) ← Detected
│   Frame 6-7-8-9-10
├─ [SCENE CHANGE] Medal spins (18% change) ← NOW DETECTED ✓
│   Frame 11-12-13-14-15
├─ [SCENE CHANGE] Energy buildup (16% change) ← NOW DETECTED ✓
│   Frame 16-17-18-19-20
├─ [SCENE CHANGE] Transformation flash (35% change) ← Detected
│   Frame 21-22-23-24-25
└─ [SCENE CHANGE] Materialization (20% change) ← NOW DETECTED ✓
    Frame 26-27-28-29-30

Result: Captured 5 scene changes, complete coverage!
```

### Visual Example

```
Yokai Watch Opening Sequence Comparison:

Previous (threshold=22):
[====Scene 1====]        [====Scene 2====]              [====Scene 3====]
  Watch closed            Jibanyan appears                Battle starts
      ↓                         ↓                              ↓
   Captured                 Captured                       Captured
   (3 major scenes)

Ultra-Dense (threshold=15):
[==S1==][==S2==][==S3==][==S4==][==S5==][==S6==][==S7==][==S8==][==S9==]
 Watch   Opens   Medal   Spins   Energy  Flash   Yokai   Pose   Effects
  ↓       ↓       ↓       ↓       ↓       ↓       ↓       ↓      ↓
All transitions captured (9 detailed scenes)
```

---

## Frames per Scene Deep Dive

### Strategy: Evenly Distributed Sampling

**Previous: 5 frames per scene**
```
Scene duration: 30 frames (1 second at 30fps)
Strategy: Skip first/last 10%, sample 5 frames from middle 80%

Timeline:
0%        20%       40%       60%       80%      100%
├─────────┼─────────┼─────────┼─────────┼─────────┤
│  Skip   │    ✓    │    ✓    │    ✓    │    ✓    │    ✓   │  Skip  │
└─────────┴─────────┴─────────┴─────────┴─────────┴────────┘
          Frame 6   Frame 12  Frame 18  Frame 24  Frame 30

Captured: 6, 12, 18, 24, 30 (5 frames)
Gaps: 6 frames between samples
```

**Ultra-Dense: 8 frames per scene**
```
Same 30-frame scene:

Timeline:
0%    10%    20%    30%    40%    50%    60%    70%    80%    90%   100%
├─────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼─────┤
│ Skip│  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │ Skip│
└─────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴─────┘
      F4     F7     F11    F14    F18    F21    F25    F28

Captured: 4, 7, 11, 14, 18, 21, 25, 28 (8 frames)
Gaps: ~3-4 frames between samples
```

### Why This Matters for Rapid Animations

**Example: Yokai transformation (30 frames, 1 second)**

```
Actual animation frames:
F1:  😐 Normal
F5:  😮 Eyes widen slightly
F10: 😯 Mouth opens
F15: 😲 Surprise peak
F20: ✨ Sparkle effects start
F25: ⚡ Energy aura visible
F30: 🔥 Full transformation

Previous (5 frames):
Captured: F6, F12, F18, F24, F30
Result:   😮   😯    ✨    ⚡    🔥
Missing:  Initial reaction (F1-F5), surprise peak (F15)

Ultra-Dense (8 frames):
Captured: F4, F7, F11, F14, F18, F21, F25, F28
Result:   😐  😮   😯    😲    ✨    ⚡    ⚡    🔥
Coverage: Complete progression from calm to transformation!
```

---

## Interval-Based Extraction

### Temporal Coverage Guarantee

Scene detection is great but might miss slow changes. Interval extraction provides baseline coverage.

**Previous: 4-second intervals**
```
20-minute episode (1200 seconds):
├─ 0s    ├─ 4s    ├─ 8s    ├─ 12s   ├─ 16s   ├─ 20s   ...
   ✓        ✓        ✓        ✓        ✓        ✓

Total interval frames: 1200 / 4 = 300 frames

Problem: 4-second gaps might miss short action sequences
Example: A 3-second yokai summon might occur entirely within a gap
```

**Ultra-Dense: 2-second intervals**
```
20-minute episode (1200 seconds):
├─ 0s ├─ 2s ├─ 4s ├─ 6s ├─ 8s ├─ 10s ├─ 12s ├─ 14s ├─ 16s ...
   ✓     ✓     ✓     ✓     ✓     ✓      ✓      ✓      ✓

Total interval frames: 1200 / 2 = 600 frames

Benefit: Maximum 2-second gap, guarantees capturing any significant moment
```

### Hybrid Mode: Best of Both Worlds

```
Scene-based:    Captures visual changes (semantic coverage)
Interval-based: Captures everything else (temporal coverage)
Combined:       Maximum comprehensive coverage

Example episode breakdown:
├─ Scene frames:    ~2,800 frames (from 350 scenes × 8 frames)
├─ Interval frames: ~  600 frames (20 min ÷ 2s)
├─ Deduplicated:    ~ -200 frames (some overlap removed)
└─ Total:           ~4,200 frames per episode

Compare to previous:
  Scene:    1,400 frames (280 scenes × 5 frames)
  Interval:   300 frames (20 min ÷ 4s)
  Total:    ~2,400 frames per episode

Improvement: +75% more frames = +75% more training data
```

---

## Resource Requirements

### Storage

```
Previous extraction:
  65 episodes × 2,400 frames × ~170KB/frame = ~26 GB

Ultra-dense extraction:
  Episodes: 214 episodes × 4,500 frames × 170KB = ~164 GB
  Movies:   4 movies × 90min × (60/2s) × 170KB = ~9 GB
  ─────────────────────────────────────────────────────
  Total:                                         ~173 GB

Recommendation: Ensure 200+ GB free space
```

### Processing Time

```
Per-episode processing stages:
├─ Video conversion:      ~45s (if not MP4)
├─ Scene detection:       ~90s (more scenes with threshold=15)
├─ Scene frame extraction: ~120s (8 frames vs 5 frames)
├─ Interval extraction:    ~40s (2s intervals)
└─ Total per episode:     ~295s ≈ 5 minutes

Total processing time:
  214 episodes × 5 min = 1,070 min ≈ 17.8 hours
  4 movies × 15 min = 60 min
  ─────────────────────────────────
  Total with 16 workers:  ~12-15 hours (parallel efficiency)

Previous: 65 episodes × 3.5 min = 227 min ≈ 3.8 hours

Note: Longer time but extracting 3.4× more files with 2× density
```

### CPU Usage

```
16 Workers:
  ├─ Scene detection:  16 parallel processes (CPU bound)
  ├─ Frame extraction: 16 parallel FFmpeg instances
  └─ Expected CPU:     ~90-95% utilization across all cores

Monitor during extraction:
  htop  # Should show 16 Python processes active
```

---

## Use Cases for Ultra-Dense Dataset

### 1. Character LoRA Training (Primary)

**Benefits:**
- **More pose variety:** 4,500 frames vs 2,400 = 87% more unique poses
- **Better expression coverage:** Captures micro-expressions and transitions
- **Action diversity:** Full coverage of special moves and attacks

**Example: Jibanyan LoRA**
```
Previous dataset: ~150 clear images of Jibanyan per season
Ultra-dense:      ~300 clear images of Jibanyan per season

Training data increase: 2× more variety
Result: Better generalization, fewer artifacts
```

### 2. Frame Interpolation Training

**Goal:** Train AI to generate smooth in-between frames

**Requirements:**
- Consecutive frames showing motion progression
- High temporal resolution (small time gaps)

**Ultra-dense advantages:**
```
Example: Character jumping motion (30 frames, 1 second)

Previous (sparse):
F0 → F12 → F24 → F30
(Ground) (Peak) (Landing)
Gap: 12 frames to interpolate

Ultra-dense:
F0 → F4 → F7 → F11 → F14 → F18 → F21 → F25 → F28 → F30
Gap: 3-4 frames to interpolate (much easier!)

Result: Can train interpolation models with real data
```

### 3. Motion Sequence Analysis

**Goal:** Understand and replicate animation patterns

**Use case:** Learn how yokai summoning animations work
```
Dataset provides:
├─ Complete summoning sequences (15-20 frames each)
├─ Energy buildup patterns (particle effects progression)
├─ Transformation stages (pose-by-pose breakdown)
└─ Attack animations (wind-up, strike, recovery)

Application:
  → Train AnimateDiff models on yokai-specific motions
  → Generate new summoning sequences for custom yokai
  → Create smooth battle animations
```

### 4. Style Transfer Training

**Goal:** Apply Yokai Watch art style to other images

**Requirements:**
- Diverse backgrounds
- Various lighting conditions
- Different art styles within the series

**Benefits:**
```
More frames = More style variety:
├─ Indoor scenes (warm lighting)
├─ Outdoor scenes (natural lighting)
├─ Battle scenes (dynamic effects)
├─ Comedy scenes (exaggerated expressions)
└─ Serious scenes (realistic shading)

Each provides different stylistic training data
```

### 5. Object Detection Training

**Goal:** Automatically identify yokai in images

**Requirements:**
- Many examples of each yokai from different angles
- Various poses and expressions
- Different lighting and backgrounds

**Ultra-dense advantages:**
```
Example: Jibanyan detection model

Previous: ~50 images per episode × 65 episodes = 3,250 images
Ultra-dense: ~100 images per episode × 214 episodes = 21,400 images

Training improvement: 6.6× more training data
Result: Much more robust detection, better generalization
```

---

## Verification Steps

After extraction completes, verify the quality:

### 1. Check Frame Count

```bash
# Count total frames
find /path/to/output -name "*.jpg" | wc -l
# Expected: ~900,000-1,000,000 frames

# Check a specific episode
ls /path/to/output/episode_001/*.jpg | wc -l
# Expected: ~4,000-5,000 frames
```

### 2. Verify Temporal Density

```bash
# Check timestamps of first 20 frames in a scene
ls -lt /path/to/output/episode_001/scene_050_*.jpg | head -20

# Look for small timestamp gaps (should see frames 0.1-0.2s apart)
```

### 3. Visual Inspection

```bash
# View a summoning sequence
# Episode 1 typically has Jibanyan summon around 5:30
ffmpeg -i input.mp4 -ss 330 -t 10 -r 10 /tmp/reference_%03d.jpg

# Compare with extracted frames from same timeframe
ls /path/to/output/episode_001/*_t330*.jpg
```

### 4. Compare with Previous Extraction

```bash
# Previous extraction directory
OLD=/path/to/yokai-watch/extracted_frames
NEW=/path/to/yokai-watch/extracted_frames_ultra_dense

# Compare episode 1
OLD_COUNT=$(ls $OLD/episode_001/*.jpg 2>/dev/null | wc -l)
NEW_COUNT=$(ls $NEW/episode_001/*.jpg 2>/dev/null | wc -l)

echo "Previous: $OLD_COUNT frames"
echo "New:      $NEW_COUNT frames"
echo "Increase: $(($NEW_COUNT - $OLD_COUNT)) frames (+$(($NEW_COUNT * 100 / $OLD_COUNT - 100))%)"
```

---

## Troubleshooting

### Issue: Extraction is slow

**Check:**
```bash
# Verify 16 workers are running
ps aux | grep "python3.*universal_frame_extractor" | wc -l
# Should show multiple processes

# Check CPU usage
htop
# Should see high CPU utilization across all cores
```

**Solutions:**
- Ensure no other heavy processes are running
- Check if disk I/O is bottleneck: `iostat -x 2`
- If disk is slow, consider extracting to faster storage

### Issue: Disk space running out

**Monitor during extraction:**
```bash
# Watch disk usage
watch -n 60 'df -h /mnt/c'

# If running low, pause and clean temp files
rm -rf /path/to/temp_converted/*.mp4
```

**Long-term solution:**
- Extract to external drive with more space
- Or extract in batches (e.g., one season at a time)

### Issue: Out of memory errors

**Symptoms:**
- Extraction crashes randomly
- System becomes unresponsive

**Solution:**
```bash
# Reduce workers
--workers 8  # Instead of 16

# Or reduce frame extraction intensity
--frames-per-scene 6  # Instead of 8
```

---

## Conclusion

Ultra-dense extraction provides **maximum coverage** for training:

✅ **Complete dataset:** All 218 files (214 episodes + 4 movies)
✅ **High temporal resolution:** 2-second intervals
✅ **Sensitive scene detection:** Threshold=15 catches subtle changes
✅ **Dense frame coverage:** 8 frames per scene
✅ **Optimal for motion capture:** Perfect for yokai summoning animations
✅ **Future-proof:** Enough data for multiple training purposes

**Trade-offs:**
- Longer processing time (12-15 hours vs 4 hours)
- More storage required (173 GB vs 26 GB)
- Higher initial complexity

**Worth it?**
Absolutely! For training robust AI models, having 2-3× more high-quality data is **much more valuable** than saving a few hours of processing time.

---

**Ready to extract?**
```bash
chmod +x scripts/batch/yokai_watch_ultra_dense.sh
bash scripts/batch/yokai_watch_ultra_dense.sh
```

The extraction will run for 12-15 hours. You can monitor progress with:
```bash
watch -n 300 'find /path/to/output -name "*.jpg" | wc -l'
```

Good luck! 🎮✨
