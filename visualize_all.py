import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

def extract_epoch_number(model_name):
    """Extract epoch number from model name for sorting"""
    if model_name.lower() == "base model":
        return -1  # Base model comes first
    
    # Look for "epoch_X" or similar patterns
    match = re.search(r'epoch[_\s]*(\d+)', model_name.lower())
    if match:
        return int(match.group(1))
    
    # Try to find just a number
    match = re.search(r'(\d+)', model_name)
    if match:
        return int(match.group(1))
    
    # If no number found, return a high number to sort it last
    return 9999

def get_display_name(filename):
    """Convert filename to a clean display name and extract epoch info"""
    # Remove file extension
    display_name = filename.replace("_results.json", "")
    
    # Handle special case for base model
    if display_name.lower() in ["base_model", "basemodel"]:
        return "Base Model"
    
    # Format epoch names consistently
    match = re.search(r'epoch[_\s]*(\d+)', display_name.lower())
    if match:
        return f"Epoch {match.group(1)}"
    
    # Make the name more readable
    display_name = display_name.replace("_", " ").title()
    return display_name

def sort_model_names(model_names):
    """Sort model names by epoch, with base model first"""
    return sorted(model_names, key=extract_epoch_number)

def analyze_results_files(results_dir="./outputs-forestfire", output_dir=None):
    """
    Analyze results JSON files and create visualizations for epoch progression
    
    Args:
        results_dir: Directory containing *_results.json files
        output_dir: Directory to save visualizations (defaults to results_dir/visualizations)
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all results files
    results_files = glob.glob(os.path.join(results_dir, "*_results.json"))
    
    if not results_files:
        print(f"No results files found in {results_dir}")
        return
    
    print(f"Found {len(results_files)} results files to analyze")
    
    # Load results and map to display names
    comparison_results = {}
    for file_path in results_files:
        try:
            # Extract model name from filename
            filename = os.path.basename(file_path)
            model_name = get_display_name(filename)
            
            # Load the results
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            comparison_results[model_name] = results
            print(f"Loaded results for {model_name}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if not comparison_results:
        print("No valid results loaded")
        return
    
    # Sort model names in chronological order (base model first, then epochs in order)
    model_names = sort_model_names(list(comparison_results.keys()))
    print(f"Models in order: {model_names}")
    
    # Define a color palette that works with sequential data
    # Use a colormap for sequential data visualization
    color_map = plt.cm.viridis
    colors = [color_map(i/max(1, len(model_names)-1)) for i in range(len(model_names))]
    
    # 1. Overall metrics comparison - Line chart showing progression over epochs
    metrics = ["accuracy", "f1_weighted", "f1_macro", "mean_sample_accuracy"]
    metric_labels = ["Accuracy", "F1 Weighted", "F1 Macro", "Mean Sample Accuracy"]
    
    plt.figure(figsize=(12, 8))
    for metric_idx, metric in enumerate(metrics):
        values = []
        for model in model_names:
            if metric in comparison_results[model]:
                values.append(comparison_results[model][metric])
            else:
                print(f"Warning: Metric '{metric}' not found for model '{model}'")
                values.append(None)  # Use None for missing data
        
        # Remove None values for plotting
        x_vals = [i for i, v in enumerate(values) if v is not None]
        y_vals = [v for v in values if v is not None]
        
        if y_vals:  # Only plot if we have values
            plt.plot(x_vals, y_vals, 'o-', linewidth=2, label=metric_labels[metric_idx])
    
    # Try to add baseline accuracy if available
    baseline_values = [comparison_results[model].get("baseline_accuracy", None) for model in model_names]
    baseline = next((b for b in baseline_values if b is not None), None)
    if baseline is not None:
        plt.axhline(y=baseline, linestyle='--', color='r', alpha=0.7, 
                   label=f'Majority Baseline ({baseline:.3f})')
    
    # Add random baseline (0.33 for 3 classes)
    plt.axhline(y=0.33, linestyle=':', color='gray', alpha=0.7, 
               label='Random Baseline (0.333)')
    
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.xlabel('Training Progress')
    plt.ylabel('Score')
    plt.title('Model Performance Progression')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "performance_progression.png"), dpi=300, bbox_inches='tight')
    print(f"Saved performance progression chart to {output_dir}/performance_progression.png")
    
    # 2. Bar chart version - for those who prefer it
    plt.figure(figsize=(14, 8))
    
    # Calculate bar positions
    num_metrics = len(metrics)
    bar_width = 0.8 / len(model_names)  # Adjust bar width based on number of models
    
    for i, model in enumerate(model_names):
        model_values = []
        for metric in metrics:
            if metric in comparison_results[model]:
                model_values.append(comparison_results[model][metric])
            else:
                model_values.append(0)
        
        # Calculate bar positions
        x = np.arange(num_metrics) + (i - len(model_names)/2 + 0.5) * bar_width
        
        plt.bar(x, model_values, bar_width, label=model, color=colors[i], alpha=0.8)
    
    # Try to add baseline accuracy if available
    if baseline is not None:
        plt.axhline(y=baseline, linestyle='--', color='r', alpha=0.7, 
                   label=f'Majority Baseline ({baseline:.3f})')
    
    # Add random baseline (0.33 for 3 classes)
    plt.axhline(y=0.33, linestyle=':', color='gray', alpha=0.7, 
               label='Random Baseline (0.333)')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(np.arange(num_metrics), metric_labels)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(model_names)//2 + 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "overall_metrics_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"Saved overall metrics comparison to {output_dir}/overall_metrics_comparison.png")
    
    # 3. Per-class F1 scores - Track improvement by class
    risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
    
    has_class_metrics = all(
        "class_metrics" in comparison_results[model] and 
        all(level in comparison_results[model]["class_metrics"] for level in risk_levels)
        for model in model_names
    )
    
    if has_class_metrics:
        # Line chart showing progression by class
        plt.figure(figsize=(12, 8))
        
        for i, level in enumerate(risk_levels):
            f1_values = []
            for model in model_names:
                try:
                    f1_values.append(comparison_results[model]["class_metrics"][level]["f1-score"])
                except KeyError:
                    print(f"Warning: Missing F1 score for {level} in {model}")
                    f1_values.append(None)
            
            # Remove None values for plotting
            x_vals = [i for i, v in enumerate(f1_values) if v is not None]
            y_vals = [v for v in f1_values if v is not None]
            
            if y_vals:  # Only plot if we have values
                plt.plot(x_vals, y_vals, 'o-', linewidth=2, label=f'{level}')
        
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.xlabel('Training Progress')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Progression by Risk Level')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "f1_progression_by_class.png"), dpi=300, bbox_inches='tight')
        print(f"Saved F1 progression by class to {output_dir}/f1_progression_by_class.png")
        
        # Bar chart for class F1 scores
        plt.figure(figsize=(14, 8))
        bar_width = 0.8 / len(model_names)
        
        for i, model in enumerate(model_names):
            class_f1 = []
            for level in risk_levels:
                try:
                    class_f1.append(comparison_results[model]["class_metrics"][level]["f1-score"])
                except KeyError:
                    class_f1.append(0)
            
            x = np.arange(len(risk_levels)) + (i - len(model_names)/2 + 0.5) * bar_width
            plt.bar(x, class_f1, bar_width, label=model, color=colors[i], alpha=0.8)
        
        plt.xlabel('Risk Level')
        plt.ylabel('F1 Score')
        plt.title('Per-Class F1 Score Comparison')
        plt.xticks(np.arange(len(risk_levels)), risk_levels)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(model_names)//2 + 1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "per_class_f1_comparison.png"), dpi=300, bbox_inches='tight')
        print(f"Saved per-class F1 comparison to {output_dir}/per_class_f1_comparison.png")
    else:
        print("Skipping per-class F1 visualization (missing class metrics)")
    
    # 4. Confusion matrices for each model
    for model in model_names:
        if "confusion_matrix" in comparison_results[model] and "confusion_matrix_labels" in comparison_results[model]:
            try:
                conf_matrix = np.array(comparison_results[model]["confusion_matrix"])
                labels = comparison_results[model]["confusion_matrix_labels"]
                
                # Create normalized confusion matrix
                row_sums = conf_matrix.sum(axis=1, keepdims=True)
                # Avoid division by zero
                row_sums = np.maximum(row_sums, 1e-10) 
                norm_conf_mx = conf_matrix / row_sums
                
                plt.figure(figsize=(10, 8))
                ax = plt.subplot(111)
                im = ax.imshow(norm_conf_mx, interpolation='nearest', cmap=plt.cm.Blues)
                plt.colorbar(im)
                
                # Set labels and ticks
                tick_marks = np.arange(len(labels))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticklabels(labels)
                
                # Add text annotations
                for i in range(conf_matrix.shape[0]):
                    for j in range(conf_matrix.shape[1]):
                        text = f"{conf_matrix[i, j]}\n({norm_conf_mx[i, j]:.2f})"
                        ax.text(j, i, text, ha="center", va="center", 
                              color="white" if norm_conf_mx[i, j] > 0.5 else "black")
                
                ax.set_title(f"Confusion Matrix - {model}")
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                plt.tight_layout()
                
                # Create a safe filename
                safe_name = model.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                
                plt.savefig(os.path.join(output_dir, f"confusion_matrix_{safe_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved confusion matrix for {model}")
            except Exception as e:
                print(f"Error creating confusion matrix for {model}: {str(e)}")
        else:
            print(f"Skipping confusion matrix for {model} (missing data)")
    
    # 5. Improvement over baseline - as a line chart to show progression
    if all("baseline_accuracy" in comparison_results[model] for model in model_names):
        try:
            improvements = []
            for model in model_names:
                improvements.append(
                    comparison_results[model]["accuracy"] - comparison_results[model]["baseline_accuracy"]
                )
            
            plt.figure(figsize=(12, 6))
            
            # Plot improvement trajectory
            plt.plot(range(len(model_names)), improvements, 'o-', linewidth=2, color='blue')
            
            # Add data points
            for i, imp in enumerate(improvements):
                plt.text(i, imp + 0.01, f"{imp:.3f}", ha='center')
            
            # Add zero line
            plt.axhline(y=0, linestyle='-', color='k', alpha=0.3)
            
            plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
            plt.xlabel('Training Progress')
            plt.ylabel('Improvement over Majority Baseline')
            plt.title('Model Improvement Trajectory')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "improvement_trajectory.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved improvement trajectory visualization")
            
            # Also create a bar chart version
            plt.figure(figsize=(12, 6))
            ax = plt.subplot(111)
            ax.bar(model_names, improvements, color=colors, alpha=0.8)
            
            # Add exact values above bars
            for i, imp in enumerate(improvements):
                ax.text(i, imp + 0.01, f"{imp:.3f}", ha='center')
            
            # Add zero line
            ax.axhline(y=0, linestyle='-', color='k', alpha=0.3)
            
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Improvement over Majority Baseline')
            plt.title('Model Improvement over Baseline')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "improvement_over_baseline.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved improvement over baseline visualization")
            
        except Exception as e:
            print(f"Error creating improvement visualization: {str(e)}")
    else:
        print("Skipping improvement over baseline visualization (missing baseline data)")
    
    # 6. Per-sample accuracy distribution
    if all("per_sample_accuracy" in comparison_results[model] for model in model_names):
        try:
            plt.figure(figsize=(12, 6))
            
            # Box plot of per-sample accuracy scores
            data = [comparison_results[model]["per_sample_accuracy"] for model in model_names]
            plt.boxplot(data, labels=model_names)
            
            # Add mean values as text
            for i, model in enumerate(model_names):
                mean_acc = comparison_results[model]["mean_sample_accuracy"]
                plt.text(i+1, -0.05, f"Mean: {mean_acc:.3f}", ha='center', va='top')
            
            plt.title('Distribution of Per-Sample Accuracy Scores')
            plt.ylabel('Proportion of Correct Predictions')
            plt.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "per_sample_accuracy_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved per-sample accuracy distribution")
        except Exception as e:
            print(f"Error creating per-sample accuracy visualization: {str(e)}")
    
    print(f"All visualizations saved to {output_dir}")
    return comparison_results

if __name__ == "__main__":
    analyze_results_files("./outputs-forestfire")