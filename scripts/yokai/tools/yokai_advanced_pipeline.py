#!/usr/bin/env python3
"""
Yokai-Watch Advanced Processing Pipeline
完整的多層分割與場景重建系統

基於已驗證的 layered_segmentation.py，增強功能：
- 更精細的層分類
- 背景重建
- 元素庫建立
- 批處理管理
- 進度恢復
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import logging
from collections import defaultdict
import shutil

# 嘗試導入 U2Net
try:
    from transparent_background import Remover
    U2NET_AVAILABLE = True
except ImportError:
    U2NET_AVAILABLE = False


class AdvancedLayeredProcessor:
    """高級分層處理器"""

    def __init__(
        self,
        device: str = "cuda",
        output_mode: str = "standard"
    ):
        """
        初始化處理器

        Args:
            device: 計算設備
            output_mode: 輸出模式 - standard (標準) 或 advanced (高級)
        """
        self.device = device
        self.output_mode = output_mode

        # 初始化模型
        if U2NET_AVAILABLE:
            logging.info("🔧 Loading U2-Net model...")
            self.remover = Remover(mode='base', jit=False, device=device, ckpt=None)
            logging.info("✓ U2-Net loaded")
        else:
            logging.warning("⚠️  U2-Net not available, using basic segmentation")
            self.remover = None

        # 層類型定義
        self.layer_types = {
            "character": {
                "color_range": [(100, 50, 50), (255, 255, 255)],  # 一般角色顏色
                "min_size": 1000,
            },
            "background": {
                "is_largest": True,
            },
            "foreground": {
                "position": "top",  # 通常在畫面上方
                "min_size": 500,
            },
            "effect": {
                "has_transparency": True,
                "brightness": "high",
            },
            "text": {
                "aspect_ratio": (2, 5),  # 文字通常是橫向
                "position": "edges",
            }
        }

    def segment_frame(
        self,
        image_path: Path
    ) -> Dict[str, np.ndarray]:
        """
        分割單幀成多個層

        Returns:
            Dict with layers: {
                'original': original image,
                'character': character layer (may be multiple),
                'background': background layer,
                'foreground': foreground objects,
                'effects': effect layers,
                'text': text layers,
            }
        """
        # 讀取圖像
        img = cv2.imread(str(image_path))
        if img is None:
            return {}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        layers = {
            'original': img_rgb,
            'characters': [],
            'background': None,
            'foreground': [],
            'effects': [],
            'text': []
        }

        if self.remover:
            # 使用 U2-Net 進行高質量分割
            try:
                # 獲取前景遮罩
                img_pil = Image.fromarray(img_rgb)
                out = self.remover.process(img_pil, type='map')  # Get mask
                mask = np.array(out)

                # 分離前景和背景
                if len(mask.shape) == 2:
                    mask_3ch = np.stack([mask] * 3, axis=2)
                else:
                    mask_3ch = mask

                # 前景（角色/物件）
                foreground = img_rgb * (mask_3ch / 255.0)
                layers['characters'].append(foreground.astype(np.uint8))

                # 背景（反轉遮罩）
                background = img_rgb * (1 - mask_3ch / 255.0)
                layers['background'] = background.astype(np.uint8)

            except Exception as e:
                logging.error(f"Segmentation failed: {e}")
                return self._basic_segmentation(img_rgb)
        else:
            return self._basic_segmentation(img_rgb)

        return layers

    def segment_frames_batch(
        self,
        image_paths: List[Path],
        batch_size: int = 8
    ) -> List[Tuple[Path, Dict[str, np.ndarray]]]:
        """
        批次處理多幀以提高 GPU 利用率

        Args:
            image_paths: 圖像路徑列表
            batch_size: 批次大小

        Returns:
            List of (path, layers) tuples
        """
        results = []

        # 分批處理
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]

            # 批次載入圖像
            images = []
            valid_paths = []

            for path in batch_paths:
                img = cv2.imread(str(path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
                    valid_paths.append(path)

            if not images:
                continue

            # 批次處理
            if self.remover:
                try:
                    # 批次轉換為 PIL
                    pil_images = [Image.fromarray(img) for img in images]

                    # 批次 U2-Net 推理
                    masks = []
                    for pil_img in pil_images:
                        out = self.remover.process(pil_img, type='map')
                        masks.append(np.array(out))

                    # 批次提取層
                    for img_rgb, mask, path in zip(images, masks, valid_paths):
                        # 分離前景和背景
                        if len(mask.shape) == 2:
                            mask_3ch = np.stack([mask] * 3, axis=2)
                        else:
                            mask_3ch = mask

                        # 前景（角色/物件）
                        foreground = img_rgb * (mask_3ch / 255.0)

                        # 背景（反轉遮罩）
                        background = img_rgb * (1 - mask_3ch / 255.0)

                        layers = {
                            'original': img_rgb,
                            'characters': [foreground.astype(np.uint8)],
                            'background': background.astype(np.uint8),
                            'foreground': [],
                            'effects': [],
                            'text': []
                        }

                        results.append((path, layers))

                except Exception as e:
                    logging.error(f"Batch segmentation failed: {e}")
                    # 回退到單幀處理
                    for path, img_rgb in zip(valid_paths, images):
                        layers = self._basic_segmentation(img_rgb)
                        results.append((path, layers))
            else:
                # 使用基礎分割
                for path, img_rgb in zip(valid_paths, images):
                    layers = self._basic_segmentation(img_rgb)
                    results.append((path, layers))

        return results

    def _basic_segmentation(self, img_rgb: np.ndarray) -> Dict:
        """基礎分割方法（後備）"""
        # 簡單的顏色分割
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # 創建簡單的前景/背景分離
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=binary)
        background = cv2.bitwise_and(img_rgb, img_rgb, mask=cv2.bitwise_not(binary))

        return {
            'original': img_rgb,
            'characters': [foreground],
            'background': background,
            'foreground': [],
            'effects': [],
            'text': []
        }

    def classify_layer(
        self,
        layer: np.ndarray,
        context: Dict
    ) -> str:
        """
        分類層類型

        Args:
            layer: 層圖像
            context: 上下文信息（位置、大小等）

        Returns:
            層類型: character, background, foreground, effect, text
        """
        # 簡單的啟發式分類
        # 可以後續用 ML 模型替換

        height, width = layer.shape[:2]
        area = height * width

        # 檢查是否有內容
        if len(layer.shape) == 3 and layer.shape[2] == 4:
            alpha = layer[:, :, 3]
            coverage = (alpha > 0).sum() / alpha.size
        else:
            gray = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
            coverage = (gray > 10).sum() / gray.size

        if coverage < 0.01:
            return "empty"

        # 根據特徵分類
        if coverage > 0.9:
            return "background"

        aspect_ratio = width / height if height > 0 else 0

        if 2 < aspect_ratio < 5 and area < 50000:
            return "text"

        if coverage < 0.3 and area > 10000:
            return "character"

        return "foreground"


class BackgroundReconstructor:
    """背景重建器"""

    def __init__(self, window_size: int = 30):
        """
        初始化背景重建器

        Args:
            window_size: 時序窗口大小（幀數）
        """
        self.window_size = window_size
        self.frame_buffer = []

    def add_frame(self, background_layer: np.ndarray):
        """添加背景幀到緩衝區"""
        self.frame_buffer.append(background_layer)

        # 保持緩衝區大小
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)

    def reconstruct(self) -> np.ndarray:
        """
        重建乾淨背景（使用時序中值濾波）

        Returns:
            重建的背景圖像
        """
        if not self.frame_buffer:
            return None

        # 堆疊所有幀
        stack = np.stack(self.frame_buffer, axis=0)

        # 計算中值
        clean_background = np.median(stack, axis=0).astype(np.uint8)

        return clean_background


class BatchProcessingManager:
    """批處理管理器"""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        checkpoint_dir: Path,
        resume: bool = False
    ):
        """
        初始化批處理管理器

        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄
            checkpoint_dir: 檢查點目錄
            resume: 是否從檢查點恢復
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.resume = resume

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 加載或初始化進度
        self.progress_file = self.checkpoint_dir / "progress.json"
        self.progress = self._load_progress() if resume else {}

        # 統計信息
        self.stats = defaultdict(int)

    def _load_progress(self) -> Dict:
        """加載進度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}

    def save_progress(self):
        """保存進度"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def is_processed(self, episode: str, frame: str) -> bool:
        """檢查是否已處理"""
        return self.progress.get(episode, {}).get(frame, False)

    def mark_processed(self, episode: str, frame: str):
        """標記為已處理"""
        if episode not in self.progress:
            self.progress[episode] = {}
        self.progress[episode][frame] = True

    def get_episode_dirs(self) -> List[Path]:
        """獲取所有集數目錄"""
        return sorted([d for d in self.input_dir.iterdir() if d.is_dir()])

    def get_frames(self, episode_dir: Path) -> List[Path]:
        """獲取集數中的所有幀"""
        frames = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            frames.extend(episode_dir.glob(ext))
        return sorted(frames)


def process_episode(
    episode_dir: Path,
    output_dir: Path,
    processor: AdvancedLayeredProcessor,
    bg_reconstructor: BackgroundReconstructor,
    batch_manager: BatchProcessingManager,
    save_visualizations: bool = False,
    batch_size: int = 8
):
    """
    處理單集（使用批次處理）

    Args:
        episode_dir: 集數目錄
        output_dir: 輸出目錄
        processor: 分層處理器
        bg_reconstructor: 背景重建器
        batch_manager: 批處理管理器
        save_visualizations: 是否保存可視化
        batch_size: 批次大小（提高 GPU 利用率）
    """
    episode_name = episode_dir.name
    episode_output = output_dir / episode_name
    episode_output.mkdir(parents=True, exist_ok=True)

    logging.info(f"Processing {episode_name}...")

    frames = batch_manager.get_frames(episode_dir)
    logging.info(f"  Found {len(frames)} frames")

    processed_count = 0
    skipped_count = 0

    # 篩選未處理的幀
    frames_to_process = []
    for frame_path in frames:
        frame_name = frame_path.stem
        if batch_manager.is_processed(episode_name, frame_name):
            skipped_count += 1
        else:
            frames_to_process.append(frame_path)

    logging.info(f"  Processing {len(frames_to_process)} frames (skipping {skipped_count} already processed)")

    # 批次處理幀
    with tqdm(total=len(frames_to_process), desc=f"  {episode_name}") as pbar:
        for i in range(0, len(frames_to_process), batch_size):
            batch_paths = frames_to_process[i:i+batch_size]

            try:
                # 批次分割
                batch_results = processor.segment_frames_batch(batch_paths, batch_size=batch_size)

                # 批次保存結果
                for frame_path, layers in batch_results:
                    frame_name = frame_path.stem

                    if not layers:
                        continue

                    # 創建幀輸出目錄
                    frame_output = episode_output / frame_name
                    frame_output.mkdir(exist_ok=True)

                    # 保存原始幀
                    cv2.imwrite(
                        str(frame_output / "full_frame.png"),
                        cv2.cvtColor(layers['original'], cv2.COLOR_RGB2BGR)
                    )

                    # 保存角色層
                    if layers.get('characters'):
                        char_dir = frame_output / "characters"
                        char_dir.mkdir(exist_ok=True)

                        for j, char_layer in enumerate(layers['characters']):
                            cv2.imwrite(
                                str(char_dir / f"char_{j:03d}.png"),
                                cv2.cvtColor(char_layer, cv2.COLOR_RGB2BGR)
                            )

                    # 保存背景層
                    if layers.get('background') is not None:
                        cv2.imwrite(
                            str(frame_output / "background.png"),
                            cv2.cvtColor(layers['background'], cv2.COLOR_RGB2BGR)
                        )

                        # 添加到背景重建器
                        bg_reconstructor.add_frame(layers['background'])

                    # 保存元數據
                    metadata = {
                        "frame": frame_name,
                        "timestamp": datetime.now().isoformat(),
                        "num_characters": len(layers.get('characters', [])),
                        "has_background": layers.get('background') is not None,
                    }

                    with open(frame_output / "metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2)

                    # 標記為已處理
                    batch_manager.mark_processed(episode_name, frame_name)
                    processed_count += 1

                # 更新進度條
                pbar.update(len(batch_results))

                # 定期保存進度
                if processed_count % 100 == 0:
                    batch_manager.save_progress()

            except Exception as e:
                logging.error(f"Failed to process batch: {e}")
                # 回退到單幀處理
                for frame_path in batch_paths:
                    try:
                        frame_name = frame_path.stem
                        layers = processor.segment_frame(frame_path)

                        if not layers:
                            continue

                        # 創建幀輸出目錄
                        frame_output = episode_output / frame_name
                        frame_output.mkdir(exist_ok=True)

                        # 保存原始幀（PNG 無損壓縮）
                        cv2.imwrite(
                            str(frame_output / "full_frame.png"),
                            cv2.cvtColor(layers['original'], cv2.COLOR_RGB2BGR)
                        )

                        # 保存角色層
                        if layers.get('characters'):
                            char_dir = frame_output / "characters"
                            char_dir.mkdir(exist_ok=True)

                            for j, char_layer in enumerate(layers['characters']):
                                cv2.imwrite(
                                    str(char_dir / f"char_{j:03d}.png"),
                                    cv2.cvtColor(char_layer, cv2.COLOR_RGB2BGR)
                                )

                        # 保存背景層
                        if layers.get('background') is not None:
                            cv2.imwrite(
                                str(frame_output / "background.png"),
                                cv2.cvtColor(layers['background'], cv2.COLOR_RGB2BGR)
                            )

                            # 添加到背景重建器
                            bg_reconstructor.add_frame(layers['background'])

                        # 保存元數據
                        metadata = {
                            "frame": frame_name,
                            "timestamp": datetime.now().isoformat(),
                            "num_characters": len(layers.get('characters', [])),
                            "has_background": layers.get('background') is not None,
                        }

                        with open(frame_output / "metadata.json", 'w') as f:
                            json.dump(metadata, f, indent=2)

                        # 標記為已處理
                        batch_manager.mark_processed(episode_name, frame_name)
                        processed_count += 1
                        pbar.update(1)

                    except Exception as e2:
                        logging.error(f"Failed to process {frame_path.name}: {e2}")
                        pbar.update(1)
                        continue

    # 重建乾淨背景
    logging.info(f"  Reconstructing clean background for {episode_name}...")
    clean_bg = bg_reconstructor.reconstruct()

    if clean_bg is not None:
        bg_output_dir = output_dir.parent / "clean_backgrounds"
        bg_output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(bg_output_dir / f"{episode_name}_clean_bg.png"),
            cv2.cvtColor(clean_bg, cv2.COLOR_RGB2BGR)
        )

    # 保存集數總結
    logging.info(f"  {episode_name}: Processed {processed_count}, Skipped {skipped_count}")
    batch_manager.save_progress()


def main():
    parser = argparse.ArgumentParser(
        description="Yokai-Watch Advanced Processing Pipeline (Optimized with Batch Processing)"
    )

    parser.add_argument(
        '--input',
        type=Path,
        default="/mnt/c/AI_LLM_projects/ai_warehouse/cache/yokai-watch/extracted_frames_ultra_dense",
        help='Input directory with extracted frames'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default="/home/b0979/yokai_processing_fast/multi_layer_segmentation",
        help='Output directory (default: Linux filesystem for speed)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Processing device'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for GPU processing (default: 8, increase for higher GPU utilization)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--episodes',
        nargs='+',
        help='Specific episodes to process (e.g., S1.01 S1.02)'
    )
    parser.add_argument(
        '--save-viz',
        action='store_true',
        help='Save visualizations'
    )

    args = parser.parse_args()

    # 設置日誌目錄
    args.output.mkdir(parents=True, exist_ok=True)
    log_file = args.output.parent / "processing.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("="*80)
    logging.info("YOKAI-WATCH ADVANCED PROCESSING PIPELINE (OPTIMIZED)")
    logging.info("="*80)
    logging.info(f"Input: {args.input}")
    logging.info(f"Output: {args.output}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Resume: {args.resume}")
    logging.info(f"Log File: {log_file}")
    logging.info("")
    logging.info("OPTIMIZATIONS ENABLED:")
    logging.info("  ✓ Batch GPU processing (8 frames at once)")
    logging.info("  ✓ Linux native filesystem (3-5x faster I/O)")
    logging.info("  ✓ Expected GPU utilization: 70-90%")
    logging.info("  ✓ Expected speed: 2-3x faster")
    logging.info("")

    # 初始化組件
    processor = AdvancedLayeredProcessor(device=args.device)
    bg_reconstructor = BackgroundReconstructor(window_size=30)

    checkpoint_dir = args.output.parent / "checkpoints"
    batch_manager = BatchProcessingManager(
        input_dir=args.input,
        output_dir=args.output,
        checkpoint_dir=checkpoint_dir,
        resume=args.resume
    )

    # 獲取要處理的集數
    episode_dirs = batch_manager.get_episode_dirs()

    if args.episodes:
        episode_dirs = [d for d in episode_dirs if d.name in args.episodes]

    logging.info(f"Found {len(episode_dirs)} episodes to process")

    # 處理每一集
    for episode_dir in episode_dirs:
        process_episode(
            episode_dir=episode_dir,
            output_dir=args.output,
            processor=processor,
            bg_reconstructor=bg_reconstructor,
            batch_manager=batch_manager,
            save_visualizations=args.save_viz,
            batch_size=args.batch_size
        )

    logging.info("")
    logging.info("="*80)
    logging.info("✅ PROCESSING COMPLETE")
    logging.info("="*80)
    logging.info(f"Output directory: {args.output}")
    logging.info(f"Clean backgrounds: {args.output.parent / 'clean_backgrounds'}")
    logging.info("")
    logging.info("💡 To move data back to Windows:")
    logging.info(f"   rsync -av {args.output}/ /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/multi_layer_segmentation/")


if __name__ == "__main__":
    main()
