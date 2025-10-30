# Advanced Tools Specification

Complete specification for all advanced Yokai Watch training tools.

**Taxonomy Version**: 2025-10 Yokai schema extended version
**Reference**: `docs/YOKAI_SCHEMA_EXTENDED.md`
**Source**: Official Yo-kai Watch game/anime locations, Fandom Wiki categories

---

## ✅ Implemented Tools

### 1. yokai_summon_scene_detector.py
**Status**: ✅ Implemented

**Purpose**: Detect spectacular yokai summon sequences with visual and audio analysis

**Features**:
- Visual effect detection (flashes, magic circles, light beams, color bursts)
- Audio integration (sound effects, music energy analysis)
- Key frame extraction
- Scene scoring system (0-100)
- Three extraction modes (key/all/sample)

**Usage**:
```bash
python3 scripts/tools/yokai_summon_scene_detector.py \
    /path/to/episodes \
    --output-dir /path/to/summon_scenes \
    --extract-mode key \
    --min-score 60.0
```

---

### 2. action_sequence_extractor.py
**Status**: ✅ Implemented

**Purpose**: Extract continuous action sequences for AnimateDiff motion LoRA training

**Features**:
- Motion detection via optical flow
- Scene consistency analysis
- Action type classification (special_effect, acceleration, entrance, etc.)
- Flexible sequence lengths (8/16/32/64 frames)
- AnimateDiff format output

**Usage**:
```bash
python3 scripts/tools/action_sequence_extractor.py \
    /path/to/episodes \
    --output-dir /path/to/sequences \
    --lengths 16 32 64 \
    --format animatediff
```

---

### 3. special_effects_organizer.py
**Status**: ✅ Implemented (Updated with extended taxonomy)

**Purpose**: Organize and categorize special effects for effect-focused LoRAs

**Features**:
- **Extended effect taxonomy** (2025-10 Yokai schema):
  - Summon/desummon (medal召喚, portal, mirapo)
  - Attack types (melee, weapon, projectile, beam, aoe, **soultimate**)
  - Inspirit/buff/status effects (妖氣, 附身, 詛咒)
  - Transformation (shadowside, godside, fusion_ritual)
  - Magic circles, seals, talismans (魔法陣, 結界, 符咒)
  - Elemental effects (fire, water, wind, lightning, ice, earth, darkness, holy, poison)
  - Energy/aura/charge effects
  - Ambient/environment (festival_night, yokai_world_mist, weather_env, speedlines)
  - UI elements (kanji cut-in, onscreen_text, comic_onomatopoeia)
  - Device effects (yo-kai watch glow, 召喚錶光, mecha_effect)
  - **Pure effect detection** (effect-only frames, no characters)
- Effect/character layer separation
- Intensity analysis with pattern detection (radial, particles, waves)
- Metadata output with `is_pure_effect` flag

**Critical Feature - Pure Effect Detection**:
Automatically detects and marks frames that contain only effects without characters. These frames are labeled as `pure_effect` and should NOT be sent to style/pose analysis to prevent training data pollution.

**Usage**:
```bash
python3 scripts/tools/special_effects_organizer.py \
    /path/to/effect_images \
    --output-dir /path/to/organized_effects \
    --separate-layers
```

**Output**:
- `by_type/{effect_type}/`: Organized by effect category
- `pure_effects/{effect_type}/`: Effect-only layers
- `combined/{effect_type}/`: Character + effect combinations
- `*.json`: Metadata with `is_pure_effect` flag and taxonomy version

---

## ✅ Additional Implemented Tools (with Extended Taxonomy)

### 4. scene_type_classifier.py
**Status**: ✅ Implemented (Updated with extended taxonomy)

**Purpose**: Classify scene types with audio-assisted analysis using Yo-kai Watch hierarchical taxonomy

**Key Features**:
- **Hierarchical scene classification** (2025-10 Yokai schema):
  - **Realm**: human_town, rural_human, resort_coastal, bbq_region, yo_kai_world, special_dimension
  - **Location**: residential_uptown, downtown_commercial, historic_hill, forest_mountain, school, harbor_beach, yo_kai_city, enma_palace, amusement_park, dungeon_tower, underground_waterway, player_home, convenience_store
  - **Environment**: indoor_home, indoor_public, outdoor_urban, outdoor_nature, outdoor_coastal, underground, fantasy_space
  - **Time**: day, night, sunset, festival_night, indoor_lit
  - **Activity**: daily_life, investigation, battle, event_festival, stealth_escape, travel
  - **Audio Environment**: quiet_residential, urban_busy, school_bell, forest_ambient, shrine_ambient, beach_wave, dungeon_echo, themepark_bgm, festival_taiko, oni_time_alarm
- Visual + audio dual analysis
- Integration with background LoRA preparation
- Scene metadata enrichment

**Algorithm**:
1. **Visual Analysis**:
   - Color histogram (warm = indoor, cool = outdoor)
   - Edge density (high = urban, low = nature)
   - Brightness distribution (dark = night, bright = day)
   - Scene structure patterns

2. **Audio Analysis** (librosa):
   - Ambient sound classification (birds = outdoor, echo = indoor)
   - Music tempo analysis (fast = battle, slow = daily)
   - Spectral features for environment detection
   - Special audio cues (taiko = festival, siren = oni_time)

3. **Hierarchical Classification**:
   - First determine realm (human vs yo-kai world)
   - Then classify location within realm
   - Determine environment type
   - Detect time of day
   - Classify activity type
   - Identify audio environment

**Expected Output**:
```json
{
  "realm": "human_town",
  "location": "school",
  "environment": "indoor_public",
  "time": "day",
  "activity": "daily_life",
  "audio_env": "school_bell",
  "confidence": {
    "realm": 0.92,
    "location": 0.85,
    "environment": 0.88
  },
  "taxonomy_version": "2025-10 Yokai schema extended"
}
```

---

### 5. yokai_style_classifier.py
**Status**: 🔧 Specification Ready

**Purpose**: Classify yokai by visual style and type for grouped training

**Key Features**:
- AI-powered classification (CLIP/BLIP2)
- Multi-dimensional categorization:
  - **Appearance**: animal, object, humanoid, abstract
  - **Attribute**: fire, water, wind, thunder, earth, light, dark
  - **Style**: cute, brave, scary, funny, cool
  - **Body Type**: quadruped, bipedal, flying, aquatic, multi-limbed
- Interactive confirmation UI
- Multi-label support (one yokai can have multiple tags)

**Classification Pipeline**:
1. **Load CLIP** model for visual embedding
2. **Extract features** from character images
3. **Compare with templates** for each category
4. **Generate classification** with confidence scores
5. **Present to user** for confirmation/adjustment
6. **Save taxonomy** for multi-concept training

**Category Templates**:
```python
CATEGORIES = {
    'appearance': {
        'animal_cat': 'a cat-like creature',
        'animal_dog': 'a dog-like creature',
        'animal_bird': 'a bird-like creature',
        'object_food': 'a food-themed character',
        'object_tool': 'a tool or weapon character',
        'humanoid': 'a humanoid character',
        ...
    },
    'attribute': {
        'fire': 'fire element, red and orange colors, flames',
        'water': 'water element, blue colors, aquatic',
        ...
    },
    ...
}
```

**Expected Usage**:
```bash
# Auto-classify with AI
python3 scripts/tools/yokai_style_classifier.py \
    /path/to/character_clusters \
    --output-json yokai_taxonomy.json \
    --auto-classify

# Interactive mode
python3 scripts/tools/yokai_style_classifier.py \
    /path/to/character_clusters \
    --output-json yokai_taxonomy.json \
    --interactive
```

---

### 6. multi_concept_lora_preparer.py
**Status**: 🔧 Specification Ready

**Purpose**: Prepare multi-character/style training data for concept LoRAs

**Key Features**:
- Group characters by style/type/attribute
- Unified trigger word system (hierarchical)
- Sample balancing across characters
- Style-specific training parameters
- Support for concept LoRA and style LoRA

**Trigger Word System**:
```
Level 1: "yokai"
Level 2: "yokai, cat-type"
Level 3: "yokai, cat-type, jibanyan"

OR

Level 1: "cute-yokai"
Level 2: "cute-yokai, cat-type"
```

**Balancing Strategy**:
- Target samples per character: 30-50 images
- Under-represented: Apply augmentation
- Over-represented: Reduce repeat count
- Equal representation in each batch

**Parameters by Group Size**:
```python
def get_params(num_characters, total_samples):
    if num_characters <= 3:
        # Few characters - learn individual + shared
        return {
            'network_dim': 64,
            'network_alpha': 32,
            'repeat': 15,
            'epochs': 20
        }
    elif num_characters <= 10:
        # Medium group - balance individual/shared
        return {
            'network_dim': 48,
            'network_alpha': 24,
            'repeat': 12,
            'epochs': 18
        }
    else:
        # Large group - focus on shared style
        return {
            'network_dim': 32,
            'network_alpha': 16,
            'repeat': 10,
            'epochs': 15
        }
```

---

### 7. interactive_style_organizer.py
**Status**: 🔧 Specification Ready

**Purpose**: Visual interface for organizing characters by style

**Key Features**:
- Thumbnail grid view
- Drag-and-drop sorting
- Multi-select operations
- Tag filtering
- Multiple classification schemes
- Export selected groups

**UI Flow**:
1. Load all character clusters
2. Display thumbnails with AI-suggested tags
3. Allow drag-drop into style categories
4. Batch operations (select multiple → apply tag)
5. Preview group composition
6. Export organization JSON

---

### 8. advanced_pose_extractor.py
**Status**: ✅ Implemented (Updated with extended taxonomy)

**Purpose**: Extract character poses for ControlNet pose training with Yo-kai Watch body type taxonomy

**Key Features**:
- **60+ Yo-kai body types** (2025-10 Yokai schema):
  - **Humanoid**: humanoid_standard, humanoid_chibi, humanoid_longlimb, oni_ogre, tengu_winged, butler_ghost
  - **Animal/Beast**: bipedal_animal, quadruped_beast, kappa_aquatic, komainu_guardian, serpentine_longbody, dragon_beast, centaur_like, avian_beast, aquatic_fishlike, insect_like, plant_rooted_beast
  - **Aerial/Floating**: flying_selfpowered, cloud_rider, floating_spirit, jellyfish_floating
  - **Complex**: multi_limb_tentacle, segmented_body, swarm_group, fusion_compound, parasite_on_host
  - **Mechanical/Object**: robot_mecha, armor_samurai, vehicle_form, object_furniture, object_tool_weapon, object_accessory_mask, object_paper_umbrella
  - **Partial**: partial_upper_body, head_only, hand_arm_only, wall_attached
  - **Soft/Liquid**: aquatic_mermaid, slime_blob, liquid_form
  - **Boss/Advanced**: giant_boss, winged_deity, shadowside_monster, godside_extended
  - **Abstract**: shadow_silhouette, energy_aura, symbolic_emblem, abstract
- OpenPose/DWPose for humanoid types
- Auto body type detection
- Special handling for non-humanoid yokai
- Sequence continuity checks
- Pose quality filtering
- Keypoint confidence scoring

**Yokai-Specific Handling**:
```python
def extract_yokai_pose(image, yokai_type):
    # Humanoid types (OpenPose)
    if yokai_type in ['humanoid_standard', 'humanoid_chibi',
                       'humanoid_longlimb', 'oni_ogre', 'butler_ghost']:
        keypoints = extract_human_pose(image)

    # Quadruped/Beast types
    elif yokai_type in ['quadruped_beast', 'komainu_guardian', 'dragon_beast']:
        keypoints = extract_animal_keypoints(image)

    # Flying/Floating types
    elif yokai_type in ['flying_selfpowered', 'cloud_rider', 'floating_spirit']:
        keypoints = extract_flying_pose(image)

    # Multi-limb types
    elif yokai_type in ['multi_limb_tentacle', 'segmented_body']:
        keypoints = extract_multi_limb_pose(image)

    # Object/Mechanical types - use bounding box + orientation
    elif 'object_' in yokai_type or 'robot_mecha' in yokai_type:
        keypoints = extract_object_pose(image)

    else:
        # Fallback to humanoid
        keypoints = extract_human_pose(image)

    return keypoints, yokai_type
```

---

### 9. anime_depth_generator.py
**Status**: 🔧 Specification Ready

**Purpose**: Generate depth maps optimized for anime style

**Key Features**:
- Anime-specific depth estimation
- Layer-based depth (character + background)
- Edge-preserving depth
- Multi-character depth ordering
- ControlNet Depth format

**Depth Generation Strategy**:
```python
def generate_anime_depth(character_layer, background_layer):
    # Background depth (furthest)
    bg_depth = estimate_background_depth(background_layer)

    # Character depth (nearest)
    char_depth = estimate_character_depth(character_layer)

    # Combine with proper ordering
    combined = np.zeros_like(bg_depth)

    # Background first (higher depth values)
    combined = bg_depth * 0.5  # Scale to 0.0-0.5

    # Character on top (lower depth values)
    char_mask = character_layer[:, :, 3] > 128
    combined[char_mask] = char_depth[char_mask] * 0.3 + 0.7  # Scale to 0.7-1.0

    return combined
```

---

### 10. controlnet_complete_pipeline.py
**Status**: 🔧 Specification Ready

**Purpose**: Generate all ControlNet conditioning images in one pass

**Key Features**:
- All control types: OpenPose, Depth, Canny, Lineart, Segmentation
- Batch processing
- Quality validation
- Automatic pairing with source images and captions
- Training-ready dataset structure

**Pipeline Flow**:
```
Input: character_clusters/
│
├─> OpenPose Extraction
├─> Depth Generation
├─> Canny Edge Detection
├─> Lineart Extraction
└─> Segmentation Mask
      │
      └─> Output:
          ├─> source/          # Original images
          ├─> pose/            # OpenPose images
          ├─> depth/           # Depth maps
          ├─> canny/           # Canny edges
          ├─> lineart/         # Line art
          ├─> segmentation/    # Segmentation masks
          └─> captions/        # Text captions
```

---

### 11. yokai_advanced_training_pipeline.sh
**Status**: 🔧 Specification Ready

**Purpose**: Integrated pipeline for all advanced training features

**Modular Design**:
```bash
# Enable specific modules
ENABLE_SUMMON_DETECTION=true
ENABLE_ACTION_SEQUENCES=true
ENABLE_EFFECTS_ORGANIZATION=true
ENABLE_STYLE_CLASSIFICATION=true
ENABLE_MULTI_CONCEPT=true
ENABLE_CONTROLNET=true

./scripts/batch/yokai_advanced_training_pipeline.sh
```

**Stages**:
1. Special scene detection (summon, effects)
2. Action sequence extraction
3. Style classification
4. Multi-concept preparation
5. ControlNet preprocessing
6. Validation
7. Generate training configs

---

### 12. YOKAI_ADVANCED_TRAINING_GUIDE.md
**Status**: 🔧 Specification Ready

**Sections**:
1. Special Scene Training
   - Summon LoRA training
   - Effect LoRA training
   - Motion LoRA (AnimateDiff) training

2. Style LoRA Training
   - Grouping characters by style
   - Multi-concept strategies
   - Trigger word design

3. ControlNet Training
   - Pose control
   - Depth control
   - Combined controls

4. Advanced Techniques
   - Combining multiple LoRAs
   - Style mixing
   - Effect injection

---

### 13. ADVANCED_TOOLS_REFERENCE.md
**Status**: 🔧 Specification Ready

**Format**: Similar to TOOLS_QUICK_REFERENCE.md but for advanced tools

**Contents**:
- All 10 new tools quick reference
- Common workflows
- Parameter tuning guide
- Integration examples

---

## 🎯 Implementation Priority

**High Priority** (Core functionality):
1. ✅ yokai_summon_scene_detector.py
2. ✅ action_sequence_extractor.py
3. ✅ special_effects_organizer.py
4. yokai_style_classifier.py
5. multi_concept_lora_preparer.py

**Medium Priority** (Enhance capabilities):
6. advanced_pose_extractor.py
7. anime_depth_generator.py
8. controlnet_complete_pipeline.py

**Low Priority** (Nice to have):
9. scene_type_classifier.py
10. interactive_style_organizer.py

**Documentation**:
11. yokai_advanced_training_pipeline.sh
12. YOKAI_ADVANCED_TRAINING_GUIDE.md
13. ADVANCED_TOOLS_REFERENCE.md

---

## 📊 Expected Impact

After full implementation:

### Training Capabilities
- ✅ Individual character LoRAs (already supported)
- ✅ Background style LoRAs (already supported)
- 🆕 Summon effect LoRAs (spectacular entrance animations)
- 🆕 Attack effect LoRAs (battle effects)
- 🆕 Style group LoRAs (all cat-type yokai, all cute yokai, etc.)
- 🆕 Motion LoRAs (AnimateDiff for animations)
- 🆕 ControlNet models (pose/depth/edge control)

### Generation Capabilities
With all LoRAs combined:
```
Base Model + Character LoRA + Effect LoRA + Style LoRA + ControlNet

Example:
"jibanyan, cute pose, summon effect, magic circle, energy aura"
  ↓
Character LoRA: Jibanyan appearance
Effect LoRA: Summon animation effects
Style LoRA: Cute yokai style
ControlNet Pose: Specific pose control
Result: Fully controlled, spectacular yokai generation
```

---

**Last Updated**: 2025-10-30
