import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os

def analyze_dataset(dataset_path: str = "forest_fire_dataset.json"):
    """
    Analyze the forest fire dataset and generate statistics.
    
    Args:
        dataset_path: Path to the JSON dataset file
    """
    print(f"Analyzing dataset: {dataset_path}")
    
    # Load dataset
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Unable to parse dataset file '{dataset_path}' as JSON.")
        return
    
    if not dataset:
        print("Error: Dataset is empty.")
        return
        
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Analyze risk level distribution
    risk_levels = ["Low Risk", "Moderate Risk", "High Risk", "Error"]
    risk_counts = {level: 0 for level in risk_levels}
    
    for item in dataset:
        answer = item.get("answer", "")
        
        # Check if it's one of the standard risk levels
        found = False
        for level in risk_levels[:3]:  # Check only the three valid risk levels
            if level == answer:
                risk_counts[level] += 1
                found = True
                break
        
        # If not a standard risk level, count as error
        if not found:
            risk_counts["Error"] += 1
    
    # Calculate percentages
    total = len(dataset)
    risk_percentages = {level: (count / total) * 100 for level, count in risk_counts.items()}
    
    # Print basic statistics
    print("\n=== RISK LEVEL DISTRIBUTION ===")
    print(f"Total samples: {total}")
    for level in risk_levels:
        print(f"{level}: {risk_counts[level]} samples ({risk_percentages[level]:.1f}%)")
    
    # Analyze sensor data
    sensor_stats = {}
    
    # First, identify what sensors we have in the first item
    if dataset and "metadata" in dataset[0] and "sensor_data" in dataset[0]["metadata"]:
        first_item = dataset[0]["metadata"]["sensor_data"]
        # Initialize stats for each sensor
        for key in first_item.keys():
            sensor_stats[key] = {
                "values": [],
                "by_risk_level": {level: [] for level in risk_levels[:3]},  # Only valid risk levels
            }
    
    # Collect sensor values
    for item in dataset:
        if "metadata" not in item or "sensor_data" not in item["metadata"]:
            continue
            
        risk_level = item.get("answer", "")
        sensor_data = item["metadata"]["sensor_data"]
        
        for key, value_str in sensor_data.items():
            # Skip if the key isn't in our stats (shouldn't happen if all entries have the same fields)
            if key not in sensor_stats:
                continue
                
            # Extract numerical value from string (remove units)
            try:
                # Handle different formats
                if "°C" in value_str:
                    value = float(value_str.replace("°C", ""))
                elif "%" in value_str:
                    value = float(value_str.replace("%", ""))
                elif "km/h" in value_str:
                    value = float(value_str.replace("km/h", ""))
                elif "mm" in value_str:
                    value = float(value_str.replace("mm", ""))
                else:
                    value = float(value_str)
                    
                sensor_stats[key]["values"].append(value)
                
                # Only add to risk level breakdown if it's a valid risk level
                if risk_level in risk_levels[:3]:
                    sensor_stats[key]["by_risk_level"][risk_level].append(value)
                    
            except ValueError:
                # Skip if we can't convert to float
                continue
    
    # Calculate statistics for each sensor
    print("\n=== SENSOR STATISTICS ===")
    for sensor, stats in sensor_stats.items():
        values = stats["values"]
        
        if not values:
            continue
            
        print(f"\n{sensor}:")
        print(f"  Range: {min(values):.2f} to {max(values):.2f}")
        print(f"  Mean: {np.mean(values):.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  Std Dev: {np.std(values):.2f}")
        
        # Print by risk level
        print(f"  By Risk Level:")
        for level in risk_levels[:3]:
            level_values = stats["by_risk_level"][level]
            if level_values:
                print(f"    {level}: Mean = {np.mean(level_values):.2f}, Range = {min(level_values):.2f} to {max(level_values):.2f}")
            else:
                print(f"    {level}: No data")
    
    # Create output directory for plots
    os.makedirs("dataset_stats", exist_ok=True)
    
    # Plot risk level distribution
    plt.figure(figsize=(10, 6))
    colors = ['green', 'orange', 'red', 'gray']
    plt.bar(risk_counts.keys(), risk_counts.values(), color=colors)
    plt.title('Forest Fire Risk Level Distribution')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count and percentage labels on top of bars
    for i, (level, count) in enumerate(risk_counts.items()):
        percentage = risk_percentages[level]
        plt.text(i, count + 0.5, f"{count}\n({percentage:.1f}%)", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("dataset_stats/risk_distribution.png")
    print("\nSaved risk distribution plot to dataset_stats/risk_distribution.png")
    
    # Plot sensor distributions by risk level
    for sensor, stats in sensor_stats.items():
        if not stats["values"]:
            continue
            
        plt.figure(figsize=(12, 6))
        
        # Clean sensor name for display
        display_name = sensor
        if "(" in sensor and ")" in sensor:
            # Extract what's inside parentheses
            display_name = sensor.split("(")[1].split(")")[0]
        
        # Create box plot
        data = [stats["by_risk_level"][level] for level in risk_levels[:3] if stats["by_risk_level"][level]]
        labels = [level for level in risk_levels[:3] if stats["by_risk_level"][level]]
        
        if not data:
            plt.close()
            continue
            
        box = plt.boxplot(data, patch_artist=True, labels=labels)
        
        # Color boxes by risk level
        for i, box_color in enumerate(colors[:len(data)]):
            box['boxes'][i].set_facecolor(box_color)
        
        plt.title(f'{display_name} Distribution by Risk Level')
        plt.ylabel(sensor)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        safe_name = sensor.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        plt.savefig(f"dataset_stats/{safe_name}_by_risk.png")
    
    print(f"\nSaved sensor distribution plots to dataset_stats/ directory")
    
    # Create correlation heatmap of sensor data
    try:
        sensor_names = list(sensor_stats.keys())
        sensor_values = []
        
        # Collect all numerical values in a matrix
        for item in dataset:
            if "metadata" not in item or "sensor_data" not in item["metadata"]:
                continue
                
            row = []
            valid_row = True
            
            for sensor in sensor_names:
                value_str = item["metadata"]["sensor_data"].get(sensor, "")
                
                try:
                    # Handle different formats
                    if "°C" in value_str:
                        value = float(value_str.replace("°C", ""))
                    elif "%" in value_str:
                        value = float(value_str.replace("%", ""))
                    elif "km/h" in value_str:
                        value = float(value_str.replace("km/h", ""))
                    elif "mm" in value_str:
                        value = float(value_str.replace("mm", ""))
                    else:
                        value = float(value_str)
                        
                    row.append(value)
                    
                except (ValueError, TypeError):
                    valid_row = False
                    break
            
            if valid_row and len(row) == len(sensor_names):
                sensor_values.append(row)
        
        if sensor_values:
            # Convert to numpy array for correlation calculation
            sensor_values = np.array(sensor_values)
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(sensor_values.T)
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add labels
            shortened_names = [name.split(' ')[0] if ' ' in name else name for name in sensor_names]
            plt.xticks(range(len(sensor_names)), shortened_names, rotation=45, ha='right')
            plt.yticks(range(len(sensor_names)), shortened_names)
            
            # Add correlation values
            for i in range(len(sensor_names)):
                for j in range(len(sensor_names)):
                    plt.text(j, i, f"{corr_matrix[i, j]:.2f}", 
                            ha='center', va='center', 
                            color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
            
            plt.colorbar(label='Correlation Coefficient')
            plt.title('Sensor Data Correlation Matrix')
            plt.tight_layout()
            plt.savefig("dataset_stats/correlation_matrix.png")
            print("Saved correlation matrix to dataset_stats/correlation_matrix.png")
    
    except Exception as e:
        print(f"Error creating correlation matrix: {str(e)}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    analyze_dataset()