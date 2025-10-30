#!/usr/bin/env python3
"""
Interactive Character Selector for Yokai Watch LoRA Training

Provides interactive UI for selecting characters/yokai for LoRA training:
- Visual cluster preview
- Filtering by size, quality, type
- Batch selection/deselection
- Export selected clusters
- Integration with cluster analysis results

Works with cluster analysis JSON to provide smart recommendations.
"""

from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Set
import json
from datetime import datetime
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class InteractiveCharacterSelector:
    """Interactive character selection interface"""

    def __init__(self, clusters_dir: Path, analysis_file: Path = None):
        """
        Initialize selector

        Args:
            clusters_dir: Directory containing character clusters
            analysis_file: Optional cluster analysis JSON
        """
        self.clusters_dir = clusters_dir
        self.analysis_file = analysis_file
        self.cluster_info = {}

        # Load cluster analysis if available
        if analysis_file and analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
                for cluster in analysis.get('clusters', []):
                    self.cluster_info[cluster['cluster_name']] = cluster

        # Find all clusters
        self.clusters = sorted([
            d for d in clusters_dir.iterdir()
            if d.is_dir() and d.name.startswith("cluster_")
        ])

        self.selected_clusters: Set[str] = set()

    def get_cluster_stats(self, cluster_dir: Path) -> Dict:
        """
        Get cluster statistics

        Args:
            cluster_dir: Cluster directory

        Returns:
            Cluster statistics
        """
        cluster_name = cluster_dir.name

        # Check if we have analysis data
        if cluster_name in self.cluster_info:
            info = self.cluster_info[cluster_name]
            return {
                "name": cluster_name,
                "num_images": info.get('num_images', 0),
                "quality": info.get('quality_score', 0),
                "diversity": info.get('diversity_score', 0),
                "tier": info.get('tier', 'Unknown'),
                "character_type": info.get('character_type', 'unknown'),
                "recommended": info.get('recommended', False),
                "has_analysis": True
            }
        else:
            # No analysis, just count images
            num_images = len(list(cluster_dir.glob("*.png")))
            return {
                "name": cluster_name,
                "num_images": num_images,
                "quality": None,
                "diversity": None,
                "tier": "Unknown",
                "character_type": "unknown",
                "recommended": num_images >= 20,
                "has_analysis": False
            }

    def get_sample_image(self, cluster_dir: Path, index: int = 0) -> Path:
        """
        Get sample image from cluster

        Args:
            cluster_dir: Cluster directory
            index: Image index

        Returns:
            Path to sample image
        """
        images = sorted(cluster_dir.glob("*.png"))
        if index < len(images):
            return images[index]
        return images[0] if images else None

    def print_cluster_info(self, cluster_dir: Path, index: int, total: int):
        """
        Print cluster information

        Args:
            cluster_dir: Cluster directory
            index: Current index
            total: Total clusters
        """
        stats = self.get_cluster_stats(cluster_dir)

        print(f"\n{'='*80}")
        print(f"Cluster {index + 1}/{total}: {stats['name']}")
        print(f"{'='*80}")

        # Basic info
        print(f"Images: {stats['num_images']}")
        print(f"Type: {stats['character_type'].title()}")

        # Quality info if available
        if stats['has_analysis']:
            print(f"Quality: {stats['quality']:.2f}")
            print(f"Diversity: {stats['diversity']:.2f}")
            print(f"Tier: {stats['tier']}")
            print(f"Recommended: {'Yes' if stats['recommended'] else 'No'}")
        else:
            print(f"Recommended: {'Yes' if stats['recommended'] else 'No'} (based on size)")

        # Selection status
        is_selected = stats['name'] in self.selected_clusters
        print(f"Status: {'[SELECTED]' if is_selected else '[NOT SELECTED]'}")

        print(f"{'='*80}")

    def show_filter_summary(self, filters: Dict):
        """
        Show active filters

        Args:
            filters: Active filters
        """
        print(f"\n📊 Active Filters:")
        if filters.get('min_images'):
            print(f"   Min images: {filters['min_images']}")
        if filters.get('max_images'):
            print(f"   Max images: {filters['max_images']}")
        if filters.get('min_quality'):
            print(f"   Min quality: {filters['min_quality']:.2f}")
        if filters.get('tiers'):
            print(f"   Tiers: {', '.join(filters['tiers'])}")
        if filters.get('types'):
            print(f"   Types: {', '.join(filters['types'])}")
        if filters.get('recommended_only'):
            print(f"   Recommended only: Yes")

    def apply_filters(self, filters: Dict) -> List[Path]:
        """
        Apply filters to clusters

        Args:
            filters: Filter criteria

        Returns:
            Filtered cluster directories
        """
        filtered = []

        for cluster_dir in self.clusters:
            stats = self.get_cluster_stats(cluster_dir)

            # Apply filters
            if filters.get('min_images') and stats['num_images'] < filters['min_images']:
                continue
            if filters.get('max_images') and stats['num_images'] > filters['max_images']:
                continue
            if filters.get('min_quality') and stats['quality'] is not None:
                if stats['quality'] < filters['min_quality']:
                    continue
            if filters.get('tiers') and stats['tier'] not in filters['tiers']:
                continue
            if filters.get('types') and stats['character_type'] not in filters['types']:
                continue
            if filters.get('recommended_only') and not stats['recommended']:
                continue

            filtered.append(cluster_dir)

        return filtered

    def interactive_selection(self):
        """
        Run interactive selection interface
        """
        print(f"\n{'='*80}")
        print("INTERACTIVE CHARACTER SELECTOR FOR YOKAI WATCH LORA")
        print(f"{'='*80}\n")

        print(f"Clusters directory: {self.clusters_dir}")
        print(f"Total clusters: {len(self.clusters)}")
        if self.analysis_file:
            print(f"Analysis file: {self.analysis_file}")
        print()

        # Current filters
        filters = {}
        current_clusters = self.clusters

        current_index = 0

        while True:
            # Show current cluster
            if not current_clusters:
                print("\n❌ No clusters match current filters")
                print("\nOptions:")
                print("  r - Reset filters")
                print("  q - Quit")
                choice = input("\nChoice: ").strip().lower()

                if choice == 'r':
                    filters = {}
                    current_clusters = self.clusters
                    current_index = 0
                    continue
                elif choice == 'q':
                    break
                continue

            cluster_dir = current_clusters[current_index]
            self.print_cluster_info(cluster_dir, current_index, len(current_clusters))

            # Show options
            print("\nOptions:")
            print("  s - Select/Deselect this cluster")
            print("  n - Next cluster")
            print("  p - Previous cluster")
            print("  j - Jump to cluster number")
            print("  v - View sample images (opens in default viewer)")
            print()
            print("  f - Apply filters")
            print("  r - Reset filters")
            print("  a - Auto-select all recommended")
            print("  c - Clear all selections")
            print()
            print(f"  x - Export selected ({len(self.selected_clusters)} clusters)")
            print("  q - Quit")

            choice = input("\nChoice: ").strip().lower()

            if choice == 's':
                # Toggle selection
                cluster_name = cluster_dir.name
                if cluster_name in self.selected_clusters:
                    self.selected_clusters.remove(cluster_name)
                    print(f"✓ Deselected {cluster_name}")
                else:
                    self.selected_clusters.add(cluster_name)
                    print(f"✓ Selected {cluster_name}")

            elif choice == 'n':
                # Next cluster
                current_index = (current_index + 1) % len(current_clusters)

            elif choice == 'p':
                # Previous cluster
                current_index = (current_index - 1) % len(current_clusters)

            elif choice == 'j':
                # Jump to cluster
                try:
                    jump_to = int(input("Cluster number (1-based): ")) - 1
                    if 0 <= jump_to < len(current_clusters):
                        current_index = jump_to
                    else:
                        print(f"Invalid cluster number (1-{len(current_clusters)})")
                except ValueError:
                    print("Invalid input")

            elif choice == 'v':
                # View sample images
                sample_images = sorted(cluster_dir.glob("*.png"))[:5]
                print(f"\nSample images (first 5):")
                for img in sample_images:
                    print(f"  {img}")

                view_choice = input("Open in viewer? (y/n): ").strip().lower()
                if view_choice == 'y':
                    for img in sample_images:
                        try:
                            Image.open(img).show()
                        except Exception as e:
                            print(f"Failed to open {img.name}: {e}")

            elif choice == 'f':
                # Apply filters
                print("\n📊 Filter Options:")
                print("  1 - Minimum images")
                print("  2 - Maximum images")
                print("  3 - Minimum quality (requires analysis)")
                print("  4 - Tiers (S/A/B/C/D)")
                print("  5 - Character type (human/yokai/unknown)")
                print("  6 - Recommended only")
                print("  0 - Apply filters")

                while True:
                    filter_choice = input("\nFilter choice (0 when done): ").strip()

                    if filter_choice == '0':
                        break
                    elif filter_choice == '1':
                        try:
                            min_imgs = int(input("Minimum images: "))
                            filters['min_images'] = min_imgs
                        except ValueError:
                            print("Invalid input")
                    elif filter_choice == '2':
                        try:
                            max_imgs = int(input("Maximum images: "))
                            filters['max_images'] = max_imgs
                        except ValueError:
                            print("Invalid input")
                    elif filter_choice == '3':
                        try:
                            min_qual = float(input("Minimum quality (0.0-1.0): "))
                            filters['min_quality'] = min_qual
                        except ValueError:
                            print("Invalid input")
                    elif filter_choice == '4':
                        tiers_input = input("Tiers (comma-separated, e.g., S,A,B): ")
                        filters['tiers'] = [t.strip().upper() for t in tiers_input.split(',')]
                    elif filter_choice == '5':
                        types_input = input("Types (comma-separated, e.g., human,yokai): ")
                        filters['types'] = [t.strip().lower() for t in types_input.split(',')]
                    elif filter_choice == '6':
                        filters['recommended_only'] = True

                # Apply filters
                current_clusters = self.apply_filters(filters)
                current_index = 0
                self.show_filter_summary(filters)

            elif choice == 'r':
                # Reset filters
                filters = {}
                current_clusters = self.clusters
                current_index = 0
                print("✓ Filters reset")

            elif choice == 'a':
                # Auto-select recommended
                for cluster_dir in current_clusters:
                    stats = self.get_cluster_stats(cluster_dir)
                    if stats['recommended']:
                        self.selected_clusters.add(stats['name'])
                print(f"✓ Auto-selected {len(self.selected_clusters)} recommended clusters")

            elif choice == 'c':
                # Clear selections
                self.selected_clusters.clear()
                print("✓ All selections cleared")

            elif choice == 'x':
                # Export
                if not self.selected_clusters:
                    print("❌ No clusters selected")
                    continue

                print(f"\n📦 Exporting {len(self.selected_clusters)} selected clusters")
                output_dir = input("Output directory: ").strip()

                if output_dir:
                    self.export_selected(Path(output_dir))

            elif choice == 'q':
                # Quit
                if self.selected_clusters:
                    save_choice = input(f"\nSave selection list? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        output_path = input("Output path (.json): ").strip()
                        if output_path:
                            self.save_selection_list(Path(output_path))
                break

        print(f"\n✓ Selection complete. Total selected: {len(self.selected_clusters)}")

    def export_selected(self, output_dir: Path):
        """
        Export selected clusters to output directory

        Args:
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting to: {output_dir}")

        for cluster_name in tqdm(self.selected_clusters, desc="Exporting clusters"):
            cluster_dir = self.clusters_dir / cluster_name
            output_cluster_dir = output_dir / cluster_name

            if cluster_dir.exists():
                shutil.copytree(cluster_dir, output_cluster_dir, dirs_exist_ok=True)

        # Save selection metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "source_dir": str(self.clusters_dir),
            "num_selected": len(self.selected_clusters),
            "selected_clusters": sorted(list(self.selected_clusters)),
            "cluster_stats": {
                cluster_name: self.get_cluster_stats(self.clusters_dir / cluster_name)
                for cluster_name in self.selected_clusters
            }
        }

        metadata_path = output_dir / "selection_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"✓ Export complete: {output_dir}")
        print(f"   Metadata: {metadata_path}")

    def save_selection_list(self, output_path: Path):
        """
        Save selection list to JSON

        Args:
            output_path: Output file path
        """
        selection_data = {
            "timestamp": datetime.now().isoformat(),
            "clusters_dir": str(self.clusters_dir),
            "num_selected": len(self.selected_clusters),
            "selected_clusters": sorted(list(self.selected_clusters))
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(selection_data, f, indent=2)

        print(f"✓ Selection list saved: {output_path}")

    def load_selection_list(self, selection_file: Path):
        """
        Load selection list from JSON

        Args:
            selection_file: Selection file path
        """
        with open(selection_file, 'r') as f:
            data = json.load(f)

        self.selected_clusters = set(data.get('selected_clusters', []))
        print(f"✓ Loaded {len(self.selected_clusters)} selections from {selection_file}")


def batch_select(
    clusters_dir: Path,
    output_dir: Path,
    analysis_file: Path = None,
    min_images: int = None,
    max_images: int = None,
    min_quality: float = None,
    tiers: List[str] = None,
    types: List[str] = None,
    recommended_only: bool = False
):
    """
    Batch selection (non-interactive)

    Args:
        clusters_dir: Clusters directory
        output_dir: Output directory
        analysis_file: Optional cluster analysis JSON
        min_images: Minimum images per cluster
        max_images: Maximum images per cluster
        min_quality: Minimum quality score
        tiers: Allowed tiers
        types: Allowed character types
        recommended_only: Only select recommended clusters
    """
    print(f"\n{'='*80}")
    print("BATCH CHARACTER SELECTION")
    print(f"{'='*80}\n")

    selector = InteractiveCharacterSelector(clusters_dir, analysis_file)

    # Build filters
    filters = {}
    if min_images:
        filters['min_images'] = min_images
    if max_images:
        filters['max_images'] = max_images
    if min_quality:
        filters['min_quality'] = min_quality
    if tiers:
        filters['tiers'] = tiers
    if types:
        filters['types'] = types
    if recommended_only:
        filters['recommended_only'] = True

    # Apply filters
    filtered_clusters = selector.apply_filters(filters)

    print(f"Matched clusters: {len(filtered_clusters)}")
    selector.show_filter_summary(filters)

    # Select all matching
    for cluster_dir in filtered_clusters:
        selector.selected_clusters.add(cluster_dir.name)

    # Export
    if selector.selected_clusters:
        selector.export_selected(output_dir)
    else:
        print("❌ No clusters matched filters")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive character selector for Yokai LoRA training"
    )

    parser.add_argument(
        "clusters_dir",
        type=Path,
        help="Directory containing character clusters"
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=None,
        help="Cluster analysis JSON file"
    )
    parser.add_argument(
        "--load-selection",
        type=Path,
        default=None,
        help="Load previous selection from JSON"
    )

    # Batch mode options
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode (non-interactive)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch mode"
    )
    parser.add_argument(
        "--min-images",
        type=int,
        help="Minimum images per cluster"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum images per cluster"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        help="Minimum quality score (0.0-1.0)"
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        help="Allowed tiers (e.g., S A B)"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        help="Allowed types (e.g., human yokai)"
    )
    parser.add_argument(
        "--recommended-only",
        action="store_true",
        help="Only select recommended clusters"
    )

    args = parser.parse_args()

    if not args.clusters_dir.exists():
        print(f"❌ Clusters directory not found: {args.clusters_dir}")
        return

    if args.batch:
        # Batch mode
        if not args.output_dir:
            print("❌ --output-dir required for batch mode")
            return

        batch_select(
            clusters_dir=args.clusters_dir,
            output_dir=args.output_dir,
            analysis_file=args.analysis,
            min_images=args.min_images,
            max_images=args.max_images,
            min_quality=args.min_quality,
            tiers=args.tiers,
            types=args.types,
            recommended_only=args.recommended_only
        )
    else:
        # Interactive mode
        selector = InteractiveCharacterSelector(
            clusters_dir=args.clusters_dir,
            analysis_file=args.analysis
        )

        if args.load_selection:
            selector.load_selection_list(args.load_selection)

        selector.interactive_selection()


if __name__ == "__main__":
    main()
