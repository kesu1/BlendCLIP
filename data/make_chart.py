import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse

def main(labels_path):
    # Predefined class names and accuracies
    class_names = ["traffic sign", "train"]
    
    top1_accuracies = [0.22719298245614036,0.6599953455899464,0.5542168674698795,0.6809222759390108,0.7248232560439017,0.8262108262108262,0.006944444444444444,0.9074074074074074,0.9553571428571429,0.7684407096171803,0.9650856389986825,0.028169014084507043,0.6102941176470589]
    
    top5_accuracies = [0.6805555555555556,0.8929485687689085,0.9322289156626506,0.9886574934920045,0.9675681020418253,1.0,0.22800925925925927,0.9891443167305236,1.0,0.9775910364145658,0.9927536231884058,0.2112676056338028,0.9779411764705882]
    
    # Read the labels file
    with open(labels_path, 'r') as f:
        lines = f.readlines()
    
    # Extract class names from each line
    classes = []
    for line in lines:
        parts = line.strip().split(':')
        if len(parts) == 2:  # Ensure we have both id and class
            class_name = parts[1].strip()
            classes.append(class_name)
    
    # Count occurrences of each class
    class_counts = Counter(classes)
    
    # Get counts for each predefined class
    ordered_counts = [class_counts.get(cls, 0) for cls in class_names]
    
    # Normalize counts
    total_count = sum(ordered_counts)
    normalized_counts = [count/total_count if total_count > 0 else 0 for count in ordered_counts]
    normalized_counts = np.array(normalized_counts)
    normalized_counts = 100 * normalized_counts  # Convert to percentage
    
    # Sort indices in descending order of class distribution
    sorted_indices = np.argsort(normalized_counts)[::-1]
    
    # Reorder everything based on sorted indices
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_normalized_counts = [normalized_counts[i] for i in sorted_indices]
    sorted_top1 = [top1_accuracies[i] * 100 for i in sorted_indices]
    sorted_top5 = [top5_accuracies[i] * 100 for i in sorted_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set the positions for bars
    x = np.arange(len(sorted_class_names))
    width = 0.3  # Width of bars
    color_top5 = '#e57373'  # light red
    color_top1 = '#b71c1c'  # dark red
    
    # Plot bars with sorted data
    ax.bar(x - width, sorted_normalized_counts, width, label='Normalized Class Distribution')
    ax.bar(x, sorted_top1, width, label='Top-1 Accuracy', color=color_top1)
    ax.bar(x + width, sorted_top5, width, label='Top-5 Accuracy', color=color_top5)
    
    # Add labels, title, and legend
    ax.set_xlabel('Classes', fontsize=16)
    ax.set_ylabel('Value (%)', fontsize=16)
    ax.set_title('Class Distribution and Classification Accuracies', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_class_names, rotation=45, ha='right', fontsize=16)
    #ax.legend()
    #ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)  # Move legend outside plot
    ax.tick_params(axis='y', labelsize=16)

    
    # Ensure the y-axis goes from 0 to 100
    ax.set_ylim(0, 100)
    
    # Add grid lines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('class_distribution_and_accuracies_sorted.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot class distribution and accuracies')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the labels file')
    args = parser.parse_args()
    
    main(args.labels_path)