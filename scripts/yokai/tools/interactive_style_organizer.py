#!/usr/bin/env python3
"""
Interactive Style Organizer

Terminal-based interactive tool for organizing characters by style:
- Browse character thumbnails (displayed as file list)
- Tag and categorize characters
- Multi-select operations
- Tag filtering
- Export organization JSON

Simplified terminal version (can be extended to GUI in future).
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Set, Optional
import json
from datetime import datetime
import shutil


class InteractiveStyleOrganizer:
    """
    Terminal-based interactive organizer for character styles
    """

    def __init__(self, clusters_dir: Path, taxonomy_file: Optional[Path] = None):
        self.clusters_dir = clusters_dir
        self.taxonomy = None

        # Load taxonomy if provided
        if taxonomy_file and taxonomy_file.exists():
            with open(taxonomy_file, 'r') as f:
                self.taxonomy = json.load(f)

        # Find all clusters
        self.clusters = sorted([
            d for d in clusters_dir.iterdir()
            if d.is_dir() and d.name.startswith('cluster_')
        ])

        # Organization state
        self.organization = {
            'groups': {},  # {group_name: [cluster_names]}
            'tags': {},    # {cluster_name: [tags]}
            'notes': {}    # {cluster_name: note_text}
        }

        # Available tag categories
        self.tag_categories = {
            'appearance': [
                'animal_cat', 'animal_dog', 'animal_bird', 'animal_dragon',
                'object_food', 'object_tool', 'humanoid', 'ghost', 'abstract'
            ],
            'attribute': [
                'fire', 'water', 'wind', 'thunder', 'earth', 'ice', 'light', 'dark', 'nature'
            ],
            'style': [
                'cute', 'cool', 'brave', 'scary', 'funny', 'mysterious', 'elegant'
            ],
            'size': [
                'tiny', 'small', 'medium', 'large', 'giant'
            ],
            'rarity': [
                'common', 'uncommon', 'rare', 'legendary'
            ]
        }

    def load_cluster_preview(self, cluster_dir: Path, max_images: int = 5) -> List[Path]:
        """
        Load preview images from cluster

        Args:
            cluster_dir: Cluster directory
            max_images: Maximum number of preview images

        Returns:
            List of image paths
        """
        image_files = sorted(cluster_dir.glob("*.png"))[:max_images]
        return image_files

    def get_cluster_stats(self, cluster_dir: Path) -> Dict:
        """
        Get cluster statistics

        Args:
            cluster_dir: Cluster directory

        Returns:
            Stats dict
        """
        image_files = list(cluster_dir.glob("*.png"))
        return {
            'num_images': len(image_files),
            'cluster_name': cluster_dir.name
        }

    def get_ai_suggestions(self, cluster_name: str) -> Optional[Dict]:
        """
        Get AI suggestions from taxonomy

        Args:
            cluster_name: Cluster name

        Returns:
            Suggested classifications or None
        """
        if not self.taxonomy:
            return None

        # Find cluster in taxonomy
        for cluster in self.taxonomy.get('clusters', []):
            if cluster.get('cluster_name') == cluster_name:
                return cluster.get('classifications', {})

        return None

    def display_cluster_info(self, cluster_dir: Path):
        """
        Display cluster information

        Args:
            cluster_dir: Cluster directory
        """
        cluster_name = cluster_dir.name
        stats = self.get_cluster_stats(cluster_dir)
        preview_images = self.load_cluster_preview(cluster_dir)

        print(f"\n{'='*80}")
        print(f"Cluster: {cluster_name}")
        print(f"{'='*80}")
        print(f"  Images: {stats['num_images']}")

        # Show preview image paths
        print(f"\n  Preview images:")
        for i, img_path in enumerate(preview_images, 1):
            print(f"    {i}. {img_path.name}")

        # Show AI suggestions if available
        ai_suggestions = self.get_ai_suggestions(cluster_name)
        if ai_suggestions:
            print(f"\n  AI Suggestions:")
            for category, labels in ai_suggestions.items():
                if labels:
                    top_label = max(labels.items(), key=lambda x: x[1])
                    print(f"    {category}: {top_label[0]} ({top_label[1]:.2f})")

        # Show current tags
        current_tags = self.organization['tags'].get(cluster_name, [])
        if current_tags:
            print(f"\n  Current tags: {', '.join(current_tags)}")

        # Show current groups
        groups_containing = [
            group_name for group_name, clusters in self.organization['groups'].items()
            if cluster_name in clusters
        ]
        if groups_containing:
            print(f"  Groups: {', '.join(groups_containing)}")

        # Show notes
        note = self.organization['notes'].get(cluster_name, '')
        if note:
            print(f"  Note: {note}")

        print(f"{'='*80}")

    def add_tags(self, cluster_name: str, tags: List[str]):
        """
        Add tags to cluster

        Args:
            cluster_name: Cluster name
            tags: List of tags to add
        """
        if cluster_name not in self.organization['tags']:
            self.organization['tags'][cluster_name] = []

        for tag in tags:
            if tag not in self.organization['tags'][cluster_name]:
                self.organization['tags'][cluster_name].append(tag)

    def remove_tags(self, cluster_name: str, tags: List[str]):
        """
        Remove tags from cluster

        Args:
            cluster_name: Cluster name
            tags: List of tags to remove
        """
        if cluster_name in self.organization['tags']:
            for tag in tags:
                if tag in self.organization['tags'][cluster_name]:
                    self.organization['tags'][cluster_name].remove(tag)

    def add_to_group(self, cluster_name: str, group_name: str):
        """
        Add cluster to group

        Args:
            cluster_name: Cluster name
            group_name: Group name
        """
        if group_name not in self.organization['groups']:
            self.organization['groups'][group_name] = []

        if cluster_name not in self.organization['groups'][group_name]:
            self.organization['groups'][group_name].append(cluster_name)

    def remove_from_group(self, cluster_name: str, group_name: str):
        """
        Remove cluster from group

        Args:
            cluster_name: Cluster name
            group_name: Group name
        """
        if group_name in self.organization['groups']:
            if cluster_name in self.organization['groups'][group_name]:
                self.organization['groups'][group_name].remove(cluster_name)

    def set_note(self, cluster_name: str, note: str):
        """
        Set note for cluster

        Args:
            cluster_name: Cluster name
            note: Note text
        """
        self.organization['notes'][cluster_name] = note

    def interactive_session(self):
        """
        Run interactive organization session
        """
        print("\n" + "="*80)
        print("INTERACTIVE STYLE ORGANIZER")
        print("="*80)
        print(f"\nClusters directory: {self.clusters_dir}")
        print(f"Total clusters: {len(self.clusters)}")
        print()

        current_index = 0

        while current_index < len(self.clusters):
            cluster_dir = self.clusters[current_index]
            cluster_name = cluster_dir.name

            # Display cluster info
            self.display_cluster_info(cluster_dir)

            # Show commands
            print("\nCommands:")
            print("  n - Next cluster")
            print("  p - Previous cluster")
            print("  t - Add/remove tags")
            print("  g - Add to group")
            print("  m - Add note")
            print("  s - Show all groups")
            print("  f - Filter clusters by tag")
            print("  j - Jump to cluster number")
            print("  q - Quit and save")
            print()

            command = input("Command: ").strip().lower()

            if command == 'n':
                current_index = min(current_index + 1, len(self.clusters) - 1)

            elif command == 'p':
                current_index = max(current_index - 1, 0)

            elif command == 't':
                print("\nAvailable tag categories:")
                for i, category in enumerate(self.tag_categories.keys(), 1):
                    print(f"  {i}. {category}")

                category_choice = input("\nChoose category (number or name): ").strip()

                # Parse category
                if category_choice.isdigit():
                    category_idx = int(category_choice) - 1
                    if 0 <= category_idx < len(self.tag_categories):
                        category = list(self.tag_categories.keys())[category_idx]
                    else:
                        print("Invalid category")
                        continue
                else:
                    category = category_choice
                    if category not in self.tag_categories:
                        print("Invalid category")
                        continue

                # Show available tags
                print(f"\nAvailable tags for {category}:")
                tags = self.tag_categories[category]
                for i, tag in enumerate(tags, 1):
                    print(f"  {i}. {tag}")

                tags_input = input("\nEnter tags to add (comma-separated) or '-' + tags to remove: ").strip()

                if tags_input.startswith('-'):
                    # Remove tags
                    tags_to_remove = [t.strip() for t in tags_input[1:].split(',')]
                    self.remove_tags(cluster_name, tags_to_remove)
                    print(f"Removed tags: {', '.join(tags_to_remove)}")
                else:
                    # Add tags
                    tags_to_add = [t.strip() for t in tags_input.split(',')]
                    self.add_tags(cluster_name, tags_to_add)
                    print(f"Added tags: {', '.join(tags_to_add)}")

            elif command == 'g':
                group_name = input("Group name: ").strip()
                self.add_to_group(cluster_name, group_name)
                print(f"Added to group: {group_name}")

            elif command == 'm':
                note = input("Note: ").strip()
                self.set_note(cluster_name, note)
                print("Note saved")

            elif command == 's':
                print("\nCurrent groups:")
                for group_name, clusters in self.organization['groups'].items():
                    print(f"\n  {group_name} ({len(clusters)} clusters):")
                    for cluster in clusters[:5]:
                        print(f"    - {cluster}")
                    if len(clusters) > 5:
                        print(f"    ... and {len(clusters) - 5} more")
                input("\nPress Enter to continue...")

            elif command == 'f':
                tag = input("Filter by tag: ").strip()
                matching = [
                    cluster_dir.name for cluster_dir in self.clusters
                    if tag in self.organization['tags'].get(cluster_dir.name, [])
                ]
                print(f"\nClusters with tag '{tag}': {len(matching)}")
                for cluster in matching[:10]:
                    print(f"  - {cluster}")
                if len(matching) > 10:
                    print(f"  ... and {len(matching) - 10} more")
                input("\nPress Enter to continue...")

            elif command == 'j':
                try:
                    jump_to = int(input("Jump to cluster number (1-{}): ".format(len(self.clusters)))) - 1
                    if 0 <= jump_to < len(self.clusters):
                        current_index = jump_to
                    else:
                        print("Invalid cluster number")
                except ValueError:
                    print("Invalid number")

            elif command == 'q':
                break

        print("\n" + "="*80)
        print("Organizing session complete")
        print("="*80)

    def export_organization(self, output_file: Path):
        """
        Export organization to JSON

        Args:
            output_file: Output JSON file
        """
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'clusters_dir': str(self.clusters_dir),
            'total_clusters': len(self.clusters),
            'organization': self.organization,
            'statistics': {
                'total_groups': len(self.organization['groups']),
                'total_tagged': len(self.organization['tags']),
                'total_with_notes': len(self.organization['notes'])
            }
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\nOrganization saved: {output_file}")

    def export_groups_as_datasets(self, output_dir: Path):
        """
        Export each group as a separate dataset directory

        Args:
            output_dir: Output base directory
        """
        print(f"\n📦 Exporting groups as datasets...")

        for group_name, cluster_names in self.organization['groups'].items():
            if not cluster_names:
                continue

            # Create group directory
            group_dir = output_dir / group_name
            group_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Group: {group_name} ({len(cluster_names)} clusters)")

            # Copy images from each cluster
            total_images = 0
            for cluster_name in cluster_names:
                cluster_dir = self.clusters_dir / cluster_name

                if not cluster_dir.exists():
                    continue

                # Copy images
                for img_file in cluster_dir.glob("*.png"):
                    dest_name = f"{cluster_name}_{img_file.name}"
                    dest_path = group_dir / dest_name
                    shutil.copy2(img_file, dest_path)
                    total_images += 1

            print(f"    Copied {total_images} images")

        print(f"\n  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive style organizer for character clusters"
    )

    parser.add_argument(
        "clusters_dir",
        type=Path,
        help="Directory containing character clusters"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=None,
        help="Optional taxonomy JSON from style classifier"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output organization JSON file"
    )
    parser.add_argument(
        "--export-groups",
        type=Path,
        default=None,
        help="Export groups as dataset directories"
    )
    parser.add_argument(
        "--load-organization",
        type=Path,
        default=None,
        help="Load existing organization JSON"
    )

    args = parser.parse_args()

    if not args.clusters_dir.exists():
        print(f"❌ Clusters directory not found: {args.clusters_dir}")
        return

    # Initialize organizer
    organizer = InteractiveStyleOrganizer(
        clusters_dir=args.clusters_dir,
        taxonomy_file=args.taxonomy
    )

    # Load existing organization if provided
    if args.load_organization and args.load_organization.exists():
        with open(args.load_organization, 'r') as f:
            loaded = json.load(f)
            organizer.organization = loaded.get('organization', {})
        print(f"Loaded organization from: {args.load_organization}")

    # Run interactive session
    organizer.interactive_session()

    # Export organization
    if args.output_json:
        organizer.export_organization(args.output_json)

    # Export groups as datasets
    if args.export_groups:
        organizer.export_groups_as_datasets(args.export_groups)

    print("\nDone!\n")


if __name__ == "__main__":
    main()
