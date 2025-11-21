"""
COCO Dataset Visualization Tool
Visualize annotations from COCO JSON format dataset.
"""

import json
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import random


class COCOVisualizer:
    """Visualize COCO format annotations."""
    
    def __init__(self, coco_json_path: str, images_dir: str):
        """
        Initialize visualizer.
        
        Args:
            coco_json_path: Path to COCO JSON annotation file
            images_dir: Directory containing the images
        """
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.coco_data = None
        self.img_to_anns = {}
        self.img_id_to_info = {}
        self.category_colors = {}
        
    def load_coco(self) -> bool:
        """Load COCO JSON file."""
        try:
            with open(self.coco_json_path, 'r') as f:
                self.coco_data = json.load(f)
            
            # Build mappings
            self._build_mappings()
            
            print(f"✓ Loaded COCO dataset:")
            print(f"  Images: {len(self.coco_data.get('images', []))}")
            print(f"  Annotations: {len(self.coco_data.get('annotations', []))}")
            print(f"  Categories: {len(self.coco_data.get('categories', []))}")
            
            return True
            
        except FileNotFoundError:
            print(f"✗ COCO JSON file not found: {self.coco_json_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON format: {e}")
            return False
    
    def _build_mappings(self):
        """Build internal mappings for fast lookup."""
        # Map image_id to annotations
        for ann in self.coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Map image_id to image info
        for img_info in self.coco_data.get('images', []):
            self.img_id_to_info[img_info['id']] = img_info
        
        # Assign colors to categories
        for cat in self.coco_data.get('categories', []):
            cat_id = cat['id']
            # Generate random color for each category
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            self.category_colors[cat_id] = color
    
    def get_category_name(self, category_id: int) -> str:
        """Get category name from ID."""
        for cat in self.coco_data.get('categories', []):
            if cat['id'] == category_id:
                return cat['name']
        return f"Unknown_{category_id}"
    
    def visualize_image(self, image_id: int = None, image_index: int = None, 
                       save_path: str = None, show: bool = True):
        """
        Visualize a single image with annotations.
        
        Args:
            image_id: COCO image ID
            image_index: Index in the images list (alternative to image_id)
            save_path: Path to save the visualization
            show: Whether to display the image
        """
        # Get image info
        if image_id is not None:
            img_info = self.img_id_to_info.get(image_id)
        elif image_index is not None:
            img_info = self.coco_data['images'][image_index]
            image_id = img_info['id']
        else:
            print("✗ Must specify either image_id or image_index")
            return
        
        if img_info is None:
            print(f"✗ Image not found")
            return
        
        # Load image
        img_path = Path(self.images_dir) / img_info['file_name']
        
        if not img_path.exists():
            print(f"✗ Image file not found: {img_path}")
            return
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"✗ Failed to load image: {img_path}")
            return
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        annotations = self.img_to_anns.get(image_id, [])
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        ax.axis('off')
        
        # Draw annotations
        for ann in annotations:
            bbox = ann.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x, y, w, h = bbox
            category_id = ann.get('category_id', 1)
            category_name = self.get_category_name(category_id)
            color = self.category_colors.get(category_id, (255, 0, 0))
            color_normalized = tuple(c / 255.0 for c in color)
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=color_normalized,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{category_name}"
            ax.text(
                x, y - 5,
                label,
                bbox=dict(facecolor=color_normalized, alpha=0.7, edgecolor='none'),
                fontsize=10,
                color='white',
                weight='bold'
            )
        
        # Add title
        title = f"{img_info['file_name']} - {len(annotations)} annotations"
        ax.set_title(title, fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_samples(self, num_samples: int = 5, save_dir: str = None, show: bool = True):
        """
        Visualize random samples from the dataset.
        
        Args:
            num_samples: Number of samples to visualize
            save_dir: Directory to save visualizations
            show: Whether to display the images
        """
        if not self.coco_data.get('images'):
            print("✗ No images in dataset")
            return
        
        # Select random images
        num_images = len(self.coco_data['images'])
        num_samples = min(num_samples, num_images)
        indices = random.sample(range(num_images), num_samples)
        
        print(f"\n→ Visualizing {num_samples} random samples...")
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for i, idx in enumerate(indices):
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f"sample_{i+1}.png"
            
            self.visualize_image(
                image_index=idx,
                save_path=save_path,
                show=show
            )
            
            if save_path:
                print(f"  [{i+1}/{num_samples}] Saved: {save_path.name}")
    
    def print_dataset_stats(self):
        """Print dataset statistics."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        # Image stats
        num_images = len(self.coco_data.get('images', []))
        print(f"\n📸 Images: {num_images}")
        
        # Annotation stats
        num_annotations = len(self.coco_data.get('annotations', []))
        print(f"📝 Annotations: {num_annotations}")
        
        if num_images > 0:
            avg_anns = num_annotations / num_images
            print(f"📊 Average annotations per image: {avg_anns:.2f}")
        
        # Category stats
        categories = self.coco_data.get('categories', [])
        print(f"\n🏷️  Categories ({len(categories)}):")
        
        # Count annotations per category
        cat_counts = {cat['id']: 0 for cat in categories}
        for ann in self.coco_data.get('annotations', []):
            cat_id = ann.get('category_id')
            if cat_id in cat_counts:
                cat_counts[cat_id] += 1
        
        for cat in categories:
            cat_id = cat['id']
            cat_name = cat['name']
            count = cat_counts.get(cat_id, 0)
            print(f"  - {cat_name}: {count} annotations")
        
        print("\n" + "="*60)


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="Visualize COCO format annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize 5 random samples from training set
  python visualize.py
  
  # Visualize 10 samples and save to directory
  python visualize.py --samples 10 --save results/visualizations/
  
  # Visualize specific dataset split
  python visualize.py --split val --samples 3
  
  # Show statistics only (no visualization)
  python visualize.py --stats-only
        """
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to visualize (default: train)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of samples to visualize (default: 5)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Directory to save visualizations'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display visualizations'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only print statistics, do not visualize'
    )
    
    args = parser.parse_args()
    
    # Determine paths based on split
    coco_json = f"dataset/{args.split}/annotations.json"
    images_dir = f"dataset/{args.split}/images"
    
    # Check if paths exist
    if not Path(coco_json).exists():
        # Try alternative structure
        coco_json = f"dataset/{args.split}/annotations.json"
        images_dir = f"dataset/{args.split}"
    
    if not Path(coco_json).exists():
        print(f"\n✗ COCO annotations not found: {coco_json}")
        print("\nPlease ensure your dataset is structured as:")
        print("  dataset/")
        print("    train/")
        print("      annotations.json")
        print("      images/")
        sys.exit(1)
    
    # Create visualizer
    print("\n" + "="*60)
    print(f"COCO DATASET VISUALIZATION - {args.split.upper()}")
    print("="*60 + "\n")
    
    visualizer = COCOVisualizer(coco_json, images_dir)
    
    if not visualizer.load_coco():
        sys.exit(1)
    
    # Print statistics
    visualizer.print_dataset_stats()
    
    # Visualize samples
    if not args.stats_only:
        show = not args.no_show
        visualizer.visualize_samples(
            num_samples=args.samples,
            save_dir=args.save,
            show=show
        )
        
        print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
