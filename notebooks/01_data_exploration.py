import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import logging

# Set up paths
BASE_DIR = r"c:\Users\chris\Downloads\FYP"
DATA_DIR = os.path.join(BASE_DIR, "dataset", "plantvillage dataset", "color")
OUTPUT_DIR = os.path.join(BASE_DIR, "notebooks", "outputs")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging for corrupted images
logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, 'corrupted_images.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def main():
    """Run exploratory data analysis on the PlantVillage dataset."""
    print("Starting EDA on PlantVillage dataset...")
    
    # 1. Scan folders and count images
    class_names = []
    class_counts = []
    image_paths_by_class = {}
    
    total_images = 0
    corrupted_count = 0
    dimensions = {}
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory not found at {DATA_DIR}")
        return

    # Gather data
    for class_name in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.isdir(class_dir):
            images = glob.glob(os.path.join(class_dir, "*.jpg")) + glob.glob(os.path.join(class_dir, "*.JPG"))
            count = len(images)
            
            class_names.append(class_name)
            class_counts.append(count)
            image_paths_by_class[class_name] = images
            total_images += count

    num_classes = len(class_names)
    print(f"Found {num_classes} classes and {total_images} total images.")

    if num_classes == 0:
        print("No classes found. Please check the dataset directory.")
        return

    # Sort for plotting
    sorted_indices = np.argsort(class_counts)[::-1]
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_class_counts = [class_counts[i] for i in sorted_indices]

    # 2. Class Distribution Plot
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    # Colors: Green if 'healthy' in name (case insensitive), else Red
    colors = ['#2ecc71' if 'healthy' in name.lower() else '#e74c3c' for name in sorted_class_names]
    
    bars = plt.bar(sorted_class_names, sorted_class_counts, color=colors)
    plt.xticks(rotation=90, ha='right', fontsize=10)
    plt.yticks(fontsize=12)
    plt.title('PlantVillage Dataset: Images per Class', fontsize=16, fontweight='bold')
    plt.xlabel('Class Name', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.tight_layout()
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Healthy'),
                       Patch(facecolor='#e74c3c', label='Diseased')]
    plt.legend(handles=legend_elements, loc='upper right', title='Status', title_fontsize='13', fontsize=12)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {os.path.join(OUTPUT_DIR, 'class_distribution.png')}")

    # 3. Sample Images Grid (6x7 = 42 slots, 38 classes)
    fig, axes = plt.subplots(6, 7, figsize=(20, 18))
    axes = axes.flatten()
    
    for i in range(42):
        ax = axes[i]
        if i < num_classes:
            cls_name = sorted_class_names[i]
            if image_paths_by_class[cls_name]:
                img_path = random.choice(image_paths_by_class[cls_name])
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(cls_name[:20] + "..." if len(cls_name) > 20 else cls_name, fontsize=8)
                except Exception as e:
                    ax.set_title("Corrupted", fontsize=8)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {os.path.join(OUTPUT_DIR, 'sample_images_grid.png')}")

    # 4 & 5. Check for corrupted images & Analyze dimensions
    print("Analyzing image dimensions and checking for corruption (this may take a while)...")
    
    for cls, paths in image_paths_by_class.items():
        for path in paths:
            try:
                with Image.open(path) as img:
                    img.verify() # Verify file integrity
                # Reopen to get actual data
                with Image.open(path) as img:
                    size = img.size # (width, height)
                    if size in dimensions:
                        dimensions[size] += 1
                    else:
                        dimensions[size] = 1
            except Exception as e:
                logging.info(f"Corrupted or invalid image: {path} - Error: {str(e)}")
                corrupted_count += 1
                
    # Save image stats
    with open(os.path.join(OUTPUT_DIR, 'image_stats.txt'), 'w') as f:
        f.write("=== Image Dimension Statistics ===\n")
        for size, count in dimensions.items():
            f.write(f"Resolution {size[0]}x{size[1]}: {count} images\n")
        f.write(f"\nTotal corrupted images found: {corrupted_count}\n")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'image_stats.txt')} and corrupted_images.txt")

    # 6. Summary Statistics
    min_imgs = min(class_counts)
    max_imgs = max(class_counts)
    mean_imgs = np.mean(class_counts)
    imbalance_ratio = max_imgs / min_imgs if min_imgs > 0 else float('inf')

    print("\n" + "="*40)
    print("=== SUMMARY STATISTICS ===")
    print("="*40)
    print(f"Total Images: {total_images}")
    print(f"Number of Classes: {num_classes}")
    print(f"Minimum Images in a Class: {min_imgs}")
    print(f"Maximum Images in a Class: {max_imgs}")
    print(f"Mean Images per Class: {mean_imgs:.2f}")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
    print(f"Total Corrupted Images: {corrupted_count}")
    print("="*40)

if __name__ == "__main__":
    main()
