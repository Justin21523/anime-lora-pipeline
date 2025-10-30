### YOKAI-WATCH EXTENDED DATA SCHEMA

*(for classifiers, dataset builders, LoRA/effects pipelines)*

**Version:** 2025-10 Yokai Extended
**Source basis:** official Yo-kai Watch game/anime locations, Fandom Wiki categories (“Yo-kai by Body Type”, “Inanimate Object Yo-kai”, “Kappa Yo-kai”, “Komainu”, “Robot Yo-kai”, “Yo-kai who can fly”, “Yo-kai riding on a cloud”), and in-anime battle/finisher VFX.

---

#### 1. Scene Taxonomy

```python
SCENE_TYPES = {
    # world / realm level
    "realm": {
        "human_town": [
            "springdale", "uptown springdale", "downtown springdale",
            "blossom heights", "shopper's row", "breezy hills",
            "sakura new town", "residential area", "shopping street"
        ],
        "rural_human": [
            "harrisville", "old springdale", "countryside", "farm area",
            "mountain village"
        ],
        "resort_coastal": [
            "san fantastico", "beach resort", "seaside", "port town"
        ],
        "bbq_region": [
            "st. peanutsburg", "bbq", "american town", "merican style street"
        ],
        "yo_kai_world": [
            "yo-kai world", "gera gera land", "paradise springs",
            "new yo-kai city", "cluvian continent", "hell's kitchen",
            "enma palace"
        ],
        "special_dimension": [
            "oni time", "terror time", "nightmare realm",
            "infinite inferno", "hooligan road", "hungry pass"
        ]
    },

    # concrete locations
    "location": {
        "residential_uptown": [
            "uptown springdale", "suburban street", "quiet housing",
            "small park near houses", "apartment front", "residential road"
        ],
        "downtown_commercial": [
            "downtown springdale", "shopping street", "convenience store front",
            "flower road", "everymart entrance", "market street", "arcade street"
        ],
        "historic_hill": [
            "blossom heights", "old houses", "shrine path", "steps to shrine",
            "cemetery on hill"
        ],
        "forest_mountain": [
            "mt. wildwood", "forest path", "sacred tree", "shrine in forest",
            "mountain trail", "river in mountain"
        ],
        "school": [
            "springdale elementary", "classroom", "school corridor",
            "schoolyard", "gym", "music room"
        ],
        "harbor_beach": [
            "beach", "pier", "fishing spot", "coast road", "seaside restaurant"
        ],
        "yo_kai_city": [
            "new yo-kai city", "yokai-world street", "floating platforms",
            "spirit market"
        ],
        "enma_palace": [
            "enma palace", "royal hall", "throne room", "demon court"
        ],
        "amusement_park": [
            "gera gera land", "theme park", "carnival", "fun rides", "stage show"
        ],
        "dungeon_tower": [
            "fuki tower", "business tower dungeon", "high-rise interior",
            "office floors", "elevator hall"
        ],
        "underground_waterway": [
            "underground waterway", "sewer", "drainage tunnel", "canal under city"
        ],
        "player_home": [
            "nate's house", "katie's house", "kid's bedroom", "living room",
            "kitchen", "bathroom"
        ],
        "convenience_store": [
            "everymart", "phantomart", "small shop", "cashier area"
        ]
    },

    # time
    "time": {
        "day": "daytime",
        "night": "nighttime",
        "sunset": "sunset or sunrise",
        "festival_night": "night with lanterns / stalls / taiko",
        "indoor_lit": "indoor with artificial light"
    },

    # activity / situation
    "activity": {
        "daily_life": ["walking", "shopping", "talking at home", "school day"],
        "investigation": ["searching for yo-kai", "using yo-kai watch", "bug catching"],
        "battle": ["yo-kai battle", "boss battle", "oni chase"],
        "event_festival": ["summer festival", "lantern festival", "dance stage"],
        "stealth_escape": ["terror time escape", "avoid oni", "run in dungeon"],
        "travel": ["train travel", "yokai elevator", "mirapo warp"]
    },

    # environment / indoor-outdoor
    "environment": {
        "indoor_home": [
            "interior", "home interior", "apartment room", "living room", "bedroom"
        ],
        "indoor_public": [
            "store interior", "school interior", "museum interior",
            "office floor", "hospital-like"
        ],
        "outdoor_urban": [
            "street scene", "shopping street", "downtown road",
            "residential street", "traffic"
        ],
        "outdoor_nature": [
            "forest", "mountain", "river bank", "park", "shrine in forest"
        ],
        "outdoor_coastal": [
            "beach", "seaside", "harbor", "port town"
        ],
        "underground": [
            "sewer", "underground waterway", "cave", "dungeon tunnel"
        ],
        "fantasy_space": [
            "yo-kai world", "floating stage", "hell's kitchen",
            "amusement park fantasy", "demon palace"
        ]
    },

    # audio environment
    "audio_env": {
        "quiet_residential": [
            "cicadas", "distant car", "light wind", "suburban ambience"
        ],
        "urban_busy": [
            "traffic", "people chatter", "shop jingles", "crosswalk beeps"
        ],
        "school_bell": [
            "school bell", "children voices", "gym echo"
        ],
        "forest_ambient": [
            "birds", "insects", "stream", "footsteps on grass"
        ],
        "shrine_ambient": [
            "wind chime", "bell strike", "light footsteps on stone"
        ],
        "beach_wave": [
            "wave", "seagull", "port machinery"
        ],
        "dungeon_echo": [
            "low reverb footsteps", "drip water", "metal gate", "wind tunnel"
        ],
        "themepark_bgm": [
            "upbeat bgm", "crowd cheer", "announcement"
        ],
        "festival_taiko": [
            "taiko drum", "festival shout", "lantern night crowd"
        ],
        "oni_time_alarm": [
            "warning siren", "heavy footsteps", "demon roar"
        ]
    }
}
```

---

#### 2. Yokai Body Types (FULL)

```python
YOKAI_BODY_TYPES = [
    # humanoid
    "humanoid_standard", "humanoid_chibi", "humanoid_longlimb",
    "oni_ogre", "tengu_winged", "butler_ghost",

    # animal / beast
    "bipedal_animal", "quadruped_beast", "kappa_aquatic",
    "komainu_guardian", "serpentine_longbody", "dragon_beast",
    "centaur_like", "avian_beast", "aquatic_fishlike",
    "insect_like", "plant_rooted_beast",

    # aerial / floating
    "flying_selfpowered", "cloud_rider", "floating_spirit", "jellyfish_floating",

    # complex
    "multi_limb_tentacle", "segmented_body", "swarm_group",
    "fusion_compound", "parasite_on_host",

    # mechanical / object
    "robot_mecha", "armor_samurai", "vehicle_form",
    "object_furniture", "object_tool_weapon",
    "object_accessory_mask", "object_paper_umbrella",

    # partial / attachment
    "partial_upper_body", "head_only", "hand_arm_only", "wall_attached",

    # soft / liquid
    "aquatic_mermaid", "slime_blob", "liquid_form",

    # boss / advanced forms
    "giant_boss", "winged_deity", "shadowside_monster", "godside_extended",

    # abstract
    "shadow_silhouette", "energy_aura", "symbolic_emblem", "abstract"
]
```

---

#### 3. Special Effects Taxonomy

```python
self.effect_types = {
    # 1) 召喚 / 出場 / 進入
    "summon": [
        "summon", "summoned", "summoning", "call", "appear", "appearance",
        "entrance", "entry", "spawn", "materialize", "teleport in",
        "yo-kai medal", "medal", "watch summon", "召喚", "召喚陣", "出現",
        "出場", "召喚光柱", "summon beam", "summon circle", "gate open", "portal",
        "mirapo", "elevator light"
    ],
    "desummon": [
        "vanish", "disappear", "teleport out", "fade out", "退場", "消失",
        "seal back", "return to medal"
    ],

    # 2) 攻擊技 (大類)
    "attack": [
        "attack", "strike", "hit", "blast", "burst", "slash", "stab",
        "shoot", "projectile", "punch", "kick", "spin attack",
        "攻擊", "斬擊", "突刺", "爆擊", "打擊", "連打"
    ],
    "attack_melee": [
        "punch", "kick", "close combat", "claw", "bite", "tail swing",
        "近戰", "近距離攻擊", "體術", "武術"
    ],
    "attack_weapon": [
        "sword", "katana", "spear", "staff", "hammer", "club", "武器攻擊",
        "刀光", "刀氣", "劍氣", "weapon slash"
    ],
    "attack_projectile": [
        "bullet", "shoot", "arrow", "wave", "energy shot", "fireball",
        "projectile", "飛彈", "飛鏢", "投射", "遠程攻擊"
    ],
    "attack_beam": [
        "beam", "laser", "ray", "kame", "straight beam", "light beam",
        "光束", "雷射", "直線光", "貫通光"
    ],
    "attack_aoe": [
        "aoe", "area attack", "shockwave", "ground slam", "地裂", "地面衝擊",
        "circle burst", "radial burst", "範圍技", "全畫面攻擊"
    ],
    "attack_soultimate": [
        "soultimate", "soultimate move", "m skill", "big move",
        "必殺", "必殺技", "大招", "big finisher", "ultimate",
        "cut-in", "kanji cut-in", "pose cut-in"
    ],

    # 3) 附身 / 妖氣 / 咒
    "inspirit": [
        "inspirit", "possession", "haunt", "curse", "debuff", "inflict",
        "妖氣", "附身", "附體", "詛咒", "降咒", "狀態異常"
    ],
    "buff_support": [
        "buff", "support", "heal", "recovery", "regeneration", "HP up",
        "defense up", "atk up", "speed up", "加速", "治癒", "治療", "回復",
        "強化", "輔助"
    ],
    "status_effect": [
        "sleep", "paralyze", "confuse", "poison", "freeze", "burning",
        "stone", "slow", "stop", "狀態", "睡眠", "麻痺", "中毒", "冰凍", "灼傷"
    ],

    # 4) 變化 / 合成 / 進化 / Shadowside
    "transformation": [
        "transform", "transformation", "evolve", "evolution", "merge",
        "fusion", "合體", "進化", "變身", "變化", "變形",
        "shadowside", "godside", "awakening", "awaken", "強化形態"
    ],
    "form_change": [
        "mode change", "armor on", "power up", "enraged form",
        "limit break", "burst mode", "超化", "模式切換"
    ],
    "fusion_ritual": [
        "fusion circle", "fusion light", "fusion seal", "合成儀式",
        "召喚儀式", "儀式光陣"
    ],

    # 5) 魔法陣 / 封印 / 結界
    "magic_circle": [
        "circle", "magic circle", "seal circle", "summon circle",
        "rune circle", "glowing circle", "地面法陣", "魔法陣", "封印陣"
    ],
    "seal_barrier": [
        "seal", "barrier", "shield", "wall", "dome", "protective ring",
        "結界", "護盾", "防護罩", "防禦結界", "六角結界"
    ],
    "talisman_ofuda": [
        "ofuda", "paper charm", "talisman", "符", "符咒", "貼符", "貼紙封印"
    ],

    # 6) 元素系 (妖怪手錶很常用的顏色爆炸)
    "elemental": [
        "fire", "water", "wind", "lightning", "ice", "earth", "dark", "light",
        "poison", "shadow", "holy", "plasma", "flame", "aqua", "storm"
    ],
    "fire": [
        "fire", "flame", "burn", "fireball", "blaze", "explosion", "炎", "火焰", "火炎", "爆炎"
    ],
    "water": [
        "water", "bubble", "splash", "wave", "tide", "aqua jet", "水", "水流", "水柱", "水彈"
    ],
    "wind": [
        "wind", "gust", "tornado", "whirlwind", "air slash", "颶風", "旋風", "風刃"
    ],
    "lightning": [
        "lightning", "thunder", "electric", "bolt", "雷擊", "電擊", "雷電柱"
    ],
    "ice": [
        "ice", "snow", "frost", "blizzard", "freeze", "凍結", "冰霜", "冰柱"
    ],
    "earth": [
        "earth", "rock", "stone", "sand", "quake", "地裂", "落石", "土石"
    ],
    "darkness": [
        "dark", "shadow", "void", "ink", "corruption", "black flame",
        "闇", "黑炎", "影子", "暗黑"
    ],
    "holy_light": [
        "light", "holy", "divine", "radiant", "blessing", "神聖", "光輝"
    ],
    "poison_miasma": [
        "poison", "toxic", "miasma", "gas cloud", "purple smoke",
        "毒霧", "瘴氣", "汙染"
    ],

    # 7) 能量 / 光束 / 光環
    "energy": [
        "energy", "aura", "power", "charge", "gathering energy",
        "energy ball", "energy burst", "能量", "氣場", "光環", "聚氣"
    ],
    "charge_up": [
        "charge", "power up", "gather", "focus", "charging", "氣功",
        "蓄力", "集氣", "吸收能量"
    ],
    "aura_mode": [
        "aura", "flame aura", "dark aura", "angel aura", "妖氣外放",
        "mode aura", "battle aura"
    ],

    # 8) 環境 / 場景特效 (LoRA 很吃這種)
    "ambient": [
        "glow", "sparkle", "shine", "soft light", "bokeh", "ambient light",
        "光暈", "閃光", "發光", "閃爍", "亮粉"
    ],
    "festival_night": [
        "lantern", "paper lantern", "matsuri", "festival lights",
        "yokai parade", "夏祭", "燈籠", "夜祭", "煙火", "花火"
    ],
    "yokai_world_mist": [
        "purple mist", "yokai world fog", "mysterious haze", "妖界霧",
        "異界紫氣", "幽冥霧"
    ],
    "weather_env": [
        "rain", "heavy rain", "snow", "sakura petals", "leaf fall",
        "rain streaks", "rain splash", "下雨", "雪花", "櫻吹雪"
    ],
    "speedlines": [
        "speedline", "motionline", "battle bg", "rapid bg", "動態線",
        "戰鬥背景", "速度線"
    ],
    "bg_burst": [
        "radial burst", "impact burst", "comic burst", "背景爆光",
        "集中線", "爆閃"
    ],

    # 9) UI / 漫畫風 / Cut-in
    "onscreen_text": [
        "kanji cut-in", "text cut-in", "big kanji", "slogan", "on-screen text",
        "必殺文字", "大字", "效果字", "招式名稱"
    ],
    "comic_onomatopoeia": [
        "bam", "pow", "boom", "zap", "ドン", "バン", "ガーン", "擬聲字"
    ],
    "frame_overlay": [
        "vignette", "edge glow", "frame fx", "UI frame", "特效框", "畫面外框"
    ],

    # 10) 裝備 / 機械 / 裝置
    "device_watch": [
        "yo-kai watch", "watch glow", "watch beam", "dial spin",
        "手錶光", "錶盤旋轉", "召喚錶光"
    ],
    "mecha_effect": [
        "mech launch", "steam", "jet", "booster", "engine glow",
        "機械噴氣", "裝甲展開", "裝備啟動"
    ],

    # 11) 純特效圖層 (沒角色時你要標這個)
    "pure_effect": [
        "fx only", "no character", "effect plate", "vfx plate",
        "empty bg with glow", "純特效", "特效圖層"
    ]
}
```
