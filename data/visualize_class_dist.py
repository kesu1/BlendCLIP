import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse

plt.rcParams["font.family"] = "Times New Roman"

def main(labels_path):
    # Predefined class names and accuracies
    class_names = [
        "barrier","pedestrian","traffic cone","construction vehicle","truck","car","bicycle","bus","motorcycle","trailer"
        ]    
    top1_accuracies = [
        0.1408557364634608,0.537064666013044,0.2956892468024633,0.34354194407456723,0.5141452758055707,0.7741687192118226,0.31153846153846154,0.786096256684492,0.5442764578833693,0.5998244844229925
        ]
    
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
    
    # Include void count in total for normalization
    void_count = class_counts.get('void', 0)
    total_count = sum(ordered_counts) + void_count
    
    # Normalize counts
    normalized_counts = [count/total_count if total_count > 0 else 0 for count in ordered_counts]
    normalized_counts = np.array(normalized_counts)
    normalized_counts = 100 * normalized_counts  # Convert to percentage
    
    # Sort indices in descending order of class distribution
    sorted_indices = np.argsort(normalized_counts)[::-1]
    
    #sorted_indices = np.argsort(top1_accuracies)[::-1]  # Sort by top-1 accuracies
    
    # Reorder everything based on sorted indices
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_normalized_counts = [normalized_counts[i] for i in sorted_indices]
    sorted_top1 = [top1_accuracies[i] * 100 for i in sorted_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set the positions for bars
    x = np.arange(len(sorted_class_names))
    width = 0.3  # Width of bars
    color_top5 = '#e57373'  # light red
    color_top1 = '#b71c1c'  # dark red
    
    highlighted_classes = []
    color_top1_highlight = "#1565c0"
    color_top5_highlight = "#64b5f6"
    
    color_top1_list = [
        color_top1_highlight if cls in highlighted_classes else color_top1
        for cls in sorted_class_names
    ]
    color_top5_list = [
        color_top5_highlight if cls in highlighted_classes else color_top5
        for cls in sorted_class_names
    ]
    
    # Plot bars with sorted data
    ax.bar(x - width/2, sorted_normalized_counts, width, label='Normalized Class Distribution', color='blue')
    ax.bar(x + width/2, sorted_top1, width, label='Top-1 Accuracy', color='red')
    #ax.bar(x + width, sorted_top5, width, label='Top-5 Accuracy', color=color_top5_list)
    #ax.bar(x - width/2, sorted_top1, width, label='Top-1 Accuracy', color=color_top1_list)
    #ax.bar(x + width/2, sorted_top5, width, label='Top-5 Accuracy', color=color_top5_list)
    
    # Add labels, title, and legend
    #ax.set_xlabel('Classes', fontsize=16)
    ax.set_ylabel('Value (%)', fontsize=18)
    #ax.set_title('Train Split Class Distribution vs. Classification Accuracies', fontsize=18)
    #ax.set_title('TruckScenes Classification Accuracies', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_class_names, rotation=45, ha='right', fontsize=18)
    #ax.legend()
    #ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)  # Move legend outside plot
    ax.tick_params(axis='y', labelsize=18)

    
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