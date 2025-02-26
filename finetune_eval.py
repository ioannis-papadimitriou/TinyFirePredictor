from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch
import numpy as np
from datasets import load_dataset, Dataset
import re
import json
import os
from transformers import TrainerCallback
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from trl import GRPOConfig, GRPOTrainer
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from collections import Counter

# Model configuration
max_seq_length = 1024
max_completion_length = 1024
max_tokens = 1024
max_prompt_length = 1024
lr = 5e-5 # 1e-4 was used initially
train_samples = 1000  # Maximum number of samples to use
val_ratio = 0.2  # 20% for validation
warmup_ratio = 0.1
lora_rank = 32
training_temperature = 0.6
model_name = "unsloth/Llama-3.2-3B-Instruct"
data_dir = "./data"  # Directory to save split datasets
save_steps = 800 # save every 1 epoch
save_total_limit = 5
epochs = 10
resume_training = True # switch to True if you want to continue from last checkpoint

SYSTEM_PROMPT = """You are an expert forest fire risk assessment system. Analyze the sensor data to determine the overall fire risk level.

Remember that forest fire risk falls into three categories:
- Low Risk: Little to no danger of fire
- Moderate Risk: Some concerning factors present
- High Risk: Significant danger of fire

Be particularly careful to distinguish Moderate Risk scenarios, which show some concerning factors but not enough for High Risk classification."""

np.random.seed(42)
torch.manual_seed(42)

class AccuracyTracker:
    """Track accuracy metrics during training with F1 estimation"""
    def __init__(self, window_size=20, risk_distribution=None):
        self.window_size = window_size
        self.history = {
            'steps': [],
            'correctness_reward': [],
            'estimated_accuracy': [],
            'f1_estimate': []
        }
        self.moving_avgs = {}
        
        # Set default risk distribution if not provided
        self.risk_distribution = risk_distribution or {"Low Risk": 0.28, "Moderate Risk": 0.25, "High Risk": 0.47}
        self.majority_class = max(self.risk_distribution, key=self.risk_distribution.get)
        self.baseline_accuracy = self.risk_distribution[self.majority_class]
        
    def update(self, step, metrics):
        # Extract correctness reward
        correctness = metrics.get('rewards/correctness_reward_func', 0)
        
        # Maximum possible reward considering class weights
        # For a batch, we would need to know the actual class distribution
        # Since that's challenging, we can use a reasonable approximation:
        max_class_reward = 2.0 * 3.0  # 2.0 base reward * max class weight (3.0)
        
        # Scale correctness to 0-100 range based on max possible reward
        estimated_accuracy = (correctness / max_class_reward) * 100
        
        # Estimate F1 based on correctness (rough approximation)
        f1_estimate = 0
        if estimated_accuracy > 0:
            # Scale F1 between random baseline (33%) and perfect accuracy
            # This still provides a reasonable trend indicator
            random_baseline = 33.3
            f1_factor = (estimated_accuracy - random_baseline) / (100 - random_baseline)
            f1_estimate = max(0, f1_factor * estimated_accuracy)
        
        # Update history and calculate moving averages as before
        self.history['steps'].append(step)
        self.history['correctness_reward'].append(correctness)
        self.history['estimated_accuracy'].append(estimated_accuracy)
        self.history['f1_estimate'].append(f1_estimate)
        
        self._calculate_moving_avgs()
        
    def _calculate_moving_avgs(self):
        for key in ['correctness_reward', 'estimated_accuracy', 'f1_estimate']:
            values = self.history[key]
            if len(values) >= self.window_size:
                self.moving_avgs[f'{key}_avg'] = sum(values[-self.window_size:]) / self.window_size
            else:
                self.moving_avgs[f'{key}_avg'] = sum(values) / len(values)
    
    def print_status(self, epoch=None):
        print("\n" + "="*50)
        if epoch is not None:
            print(f"MODEL ACCURACY - EPOCH {epoch:.2f}")
        else:
            print(f"MODEL ACCURACY - STEP {self.history['steps'][-1]}")
        print("="*50)
        
        correctness_avg = self.moving_avgs.get('correctness_reward_avg', 0)
        accuracy_avg = self.moving_avgs.get('estimated_accuracy_avg', 0)
        f1_avg = self.moving_avgs.get('f1_estimate_avg', 0)
        
        print(f"Correctness Reward: {correctness_avg:.4f}")
        print(f"Estimated Accuracy: {accuracy_avg:.1f}% (Baseline: {self.baseline_accuracy*100:.1f}%)")
        print(f"Estimated F1 Score: {f1_avg:.1f}%")
        
        # Compare to baseline
        vs_baseline = accuracy_avg - (self.baseline_accuracy * 100)
        print(f"vs. Majority Baseline: {vs_baseline:+.1f}%")
        
        if len(self.history['estimated_accuracy']) > self.window_size:
            first_window = sum(self.history['estimated_accuracy'][:self.window_size]) / self.window_size
            improvement = accuracy_avg - first_window
            print(f"\nAccuracy Improvement: {improvement:.1f}%")
        
        print("="*50 + "\n")
    
    def plot_progress(self, save_path=None):
        try:            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot correctness reward as primary metric
            ax.plot(self.history['steps'], self.history['correctness_reward'], 
                'b-', alpha=0.3, label='Raw Correctness Reward')
            
            # Add a horizontal line for maximum possible reward (approximate)
            max_reward = 2.0 * 3.0  # Based on reward function (2.0 * max class weight 3.0)
            ax.axhline(y=max_reward, color='green', linestyle='--', 
                    label=f'Max Reward ({max_reward:.1f})')
            
            # Add a horizontal line for minimum reward
            ax.axhline(y=0, color='red', linestyle='--', label='Min Reward (0.0)')
            
            if len(self.history['steps']) >= self.window_size:
                reward_avgs = []
                for i in range(self.window_size-1, len(self.history['steps'])):
                    reward_avgs.append(
                        sum(self.history['correctness_reward'][i-self.window_size+1:i+1]) / self.window_size
                    )
                ma_steps = self.history['steps'][self.window_size-1:]
                ax.plot(ma_steps, reward_avgs, 'b-', linewidth=2, 
                    label=f'Correctness Reward ({self.window_size}-pt Moving Avg)')
            
            ax.set_title('Forest Fire Risk Assessment Training Progress')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Correctness Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits with some padding
            y_max = max(self.history['correctness_reward']) * 1.1 if self.history['correctness_reward'] else max_reward * 1.1
            ax.set_ylim(0, max(y_max, max_reward * 1.1))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Training progress plot saved to {save_path}")
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for plotting. Install with: pip install matplotlib")

class AccuracyCallback(TrainerCallback):
    """Callback to track accuracy during training"""
    def __init__(self, tracker):
        self.tracker = tracker
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.tracker.update(state.global_step, logs)
            if state.global_step % 20 == 0:
                self.tracker.print_status(epoch=state.epoch)
            if state.global_step % 100 == 0 and state.global_step > 0:
                self.tracker.plot_progress(save_path=f"./outputs-forestfire/accuracy_step_{state.global_step}.png")

def extract_xml_answer(text: str) -> str:
    """Extract final answer from response text"""
    if "<answer>" in text and "</answer>" in text:
        answer_text = text.split("<answer>")[1].split("</answer>")[0].strip()
    elif "<answer>" in text:
        answer_text = text.split("<answer>")[1].strip()
    else:
        return ""
    
    # Check for risk levels in the answer text
    risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
    for level in risk_levels:
        if level.lower() in answer_text.lower():
            return level
    
    return ""

def build_prompt_from_sensor_data(sensor_data):
    """Build a prompt with just the sensor data, no guidelines"""
    sensor_text = "\n".join([f"- {key}: {value}" for key, value in sensor_data.items()])
    
    prompt = f"""Analyze the following forest sensor readings and determine the fire risk level.

Respond with your analysis followed by your conclusion in this format:

<reasoning>
[Your step-by-step analysis]
</reasoning>
<answer>
[Low Risk, Moderate Risk, or High Risk]
</answer>

Sensor readings:

{sensor_text}

What is the forest fire risk level?"""
    return prompt

def prepare_forest_fire_dataset(dataset_path="forest_fire_dataset.json", max_samples=None):
    """Prepare forest fire dataset for GRPO training, using only sensor data (no guidelines)"""
    try:
        # Load the dataset
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
        
        # Limit samples if needed
        if max_samples and max_samples < len(raw_data):
            raw_data = raw_data[:max_samples]
        
        processed_data = []
        for item in raw_data:
            # Extract the sensor data from metadata
            if "metadata" not in item or "sensor_data" not in item["metadata"]:
                print("Skipping invalid dataset entry (missing sensor data)")
                continue
                
            if "answer" not in item or not item["answer"].endswith("Risk"):
                print(f"Skipping entry with invalid answer: {item.get('answer', 'None')}")
                continue
            
            # Use only the sensor data to build a new prompt without guidelines
            sensor_data = item["metadata"]["sensor_data"]
            new_prompt = build_prompt_from_sensor_data(sensor_data)
            
            # Create chat format prompt
            prompt = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': new_prompt}
            ]
            
            processed_data.append({
                'prompt': prompt,
                'answer': item['answer']
            })
        
        # Create a dataset from the processed data
        return Dataset.from_list(processed_data)
        
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return None

def split_and_save_dataset(dataset, val_ratio=0.2, save_dir="./data"):
    """Split dataset into train/test sets and save them to disk"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Shuffle dataset with fixed seed for reproducibility
    shuffled_dataset = dataset.shuffle(seed=42)
    
    # Calculate split sizes
    val_size = max(int(len(shuffled_dataset) * val_ratio), 1)  # At least 1 validation sample
    train_size = len(shuffled_dataset) - val_size
    
    # Split dataset
    train_dataset = shuffled_dataset.select(range(train_size))
    val_dataset = shuffled_dataset.select(range(train_size, len(shuffled_dataset)))
    
    # Save datasets to disk
    train_path = os.path.join(save_dir, "train_dataset.json")
    val_path = os.path.join(save_dir, "test_dataset.json")
    
    # Convert to lists for JSON serialization
    train_data = train_dataset.to_dict()
    val_data = val_dataset.to_dict()
    
    # Save as JSON
    with open(train_path, 'w') as f:
        json.dump(train_data, f)
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f)
    
    print(f"Saved {len(train_dataset)} training samples to {train_path}")
    print(f"Saved {len(val_dataset)} test samples to {val_path}")
    
    return train_dataset, val_dataset

def load_dataset_splits(data_dir="./data"):
    """Load train and test datasets from saved files"""
    train_path = os.path.join(data_dir, "train_dataset.json")
    test_path = os.path.join(data_dir, "test_dataset.json")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Dataset splits not found. Please run split_and_save_dataset first.")
        return None, None
    
    # Load datasets from disk
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    # Convert dictionaries back to Dataset objects
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)
    
    print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    return train_dataset, test_dataset

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # Class weights to emphasize Moderate Risk
    class_weights = {
        "Low Risk": 2.0,
        "Moderate Risk": 3.0,  # Higher weights for underrepresented classes
        "High Risk": 1.0
    }
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    print('-'*20)
    print(f"Sample Prompt: {prompts[0][-1]['content']}...")
    print(f"True risk level: {answer[0]}")
    print(f"Model response: {responses[0]}...")
    print(f"Extracted answer: {extracted_responses}")

    rewards = []
    for r, a in zip(extracted_responses, answer):
        # Apply class weight to correct predictions
        if r == a:
            rewards.append(2.0 * class_weights.get(a, 1.0))
        else:
            rewards.append(0.0)
    return rewards

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for using the correct reasoning and answer format"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward for correct XML tag usage"""
    def count_xml(text):
        count = 0.0
        if text.count("<reasoning>") == 1:
            count += 0.125
        if text.count("</reasoning>") == 1:
            count += 0.125
        if text.count("<answer>") == 1:
            count += 0.125
        if text.count("</answer>") == 1:
            count += 0.125
        return count
    
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def risk_level_reward_func(completions, **kwargs) -> list[float]:
    """Reward for predicting a valid risk level"""
    valid_levels = ["Low Risk", "Moderate Risk", "High Risk"]
    extracted_answers = [extract_xml_answer(c[0]["content"]) for c in completions]
    return [0.5 if answer in valid_levels else 0.0 for answer in extracted_answers]

def display_dataset_sample(dataset, num_samples=2):
    """Display a few samples from the processed dataset"""
    if not dataset or len(dataset) == 0:
        print("Dataset is empty")
        return
        
    print("\n" + "="*80)
    print("DATASET SAMPLE PREVIEW")
    print("="*80)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"System: {sample['prompt'][0]['content']}")
        print(f"User: {sample['prompt'][1]['content'][:200]}...")
        print(f"Expected Answer: {sample['answer']}")
        print("-"*80)

def calculate_sureness_index(predictions_list):
    """
    Calculate a sureness index based on prediction consistency
    
    Args:
        predictions_list: List of model predictions for the same prompt
    
    Returns:
        float: Sureness index between 0 and 1, where 1 means all predictions were identical
    """
    if not predictions_list:
        return 0.0
    
    # Count occurrences of each prediction
    counter = Counter(predictions_list)
    
    # Calculate the proportion of the most common prediction
    most_common = counter.most_common(1)[0][1]
    sureness = most_common / len(predictions_list)
    
    return sureness

def evaluate_model(model, tokenizer, test_dataset, results_file=None, num_samples=4):
    """Evaluate a model on test dataset with detailed metrics using multiple samples per prompt"""
    
    # Ensure output directory exists if results_file is provided
    if results_file:
        os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
    
    print("\n" + "="*60)
    print(f"STARTING MODEL EVALUATION (with {num_samples} samples per prompt)")
    print("="*60)
    
    # Collect predictions for detailed metrics
    print("Collecting model predictions...")
    y_true = []
    y_pred = []
    accuracy_scores = []
    sureness_scores = []
    
    # Process each test sample to get predictions
    for i, item in enumerate(test_dataset):
        prompt = item['prompt']
        true_answer = item['answer']
        
        # Generate multiple predictions for the same prompt
        prompt_predictions = []
        for _ in range(num_samples):
            # Generate a prediction using the model
            inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                inputs, 
                max_new_tokens=max_completion_length,
                temperature=0.7,  # Higher temperature for more diverse outputs
                do_sample=True,   # Must use sampling for diversity
                top_p=0.9,        # Nucleus sampling
            )
            
            # Extract the model's response
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            predicted_answer = extract_xml_answer(response)
            prompt_predictions.append(predicted_answer if predicted_answer else "Unknown")
        
        # Use majority voting to determine final prediction
        prediction_counts = Counter(prompt_predictions)
        majority_prediction = prediction_counts.most_common(1)[0][0]

        # Calculate sureness score for this prompt
        sureness_score = calculate_sureness_index(prompt_predictions)
        sureness_scores.append(sureness_score)

        # Calculate proportion of correct predictions
        correct_count = prediction_counts.get(true_answer, 0)
        total_predictions = len(prompt_predictions)
        accuracy_score = correct_count / total_predictions
        
        print(f'Predicted: {prompt_predictions}, Actual: {true_answer}')
        # For debugging/analysis, show distribution of predictions
        if i % 10 == 0:
            print(f"\nPrompt {i} predictions distribution: {dict(prediction_counts)}")
            print(f"True label: {true_answer}, Correct predictions: {correct_count}/{total_predictions}")
            print(f"Accuracy score: {accuracy_score:.2f}")
            print(f"Majority vote: {majority_prediction}")
            print("\n" + "="*60)
            print(f"Mean Sample Accuracy Score: {sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0:.2f}")
            print(f"Mean Sureness Score: {sum(sureness_scores) / len(sureness_scores) if sureness_scores else 0:.2f}")
            print("="*60)
                
        # Store true and predicted labels
        y_true.append(true_answer)
        y_pred.append(majority_prediction)
        # Store accuracy scores for additional analysis
        accuracy_scores.append(accuracy_score)
        
        # Print progress
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(test_dataset)} test samples")
    
    # Calculate metrics
    # Define risk levels for classification report
    risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
    
    # Calculate baseline accuracy (always predicting the majority class)
    risk_distribution = {"Low Risk": 0.28, "Moderate Risk": 0.25, "High Risk": 0.47}
    majority_class = max(risk_distribution, key=risk_distribution.get)
    baseline_accuracy = risk_distribution[majority_class]
    
    # Count valid predictions (excluding unknowns)
    valid_predictions = sum(1 for pred in y_pred if pred in risk_levels)
    valid_correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    
    # Calculate accuracy
    accuracy = valid_correct / len(y_true) if len(y_true) > 0 else 0
    
    # Calculate F1 scores
    # Handle unknowns by treating them as their own class for metrics
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Get per-class F1 scores
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=risk_levels + ["Unknown"])
    
    # Prepare detailed metrics
    detailed_metrics = {
        "accuracy": accuracy,
        "accuracy_vs_baseline": accuracy - baseline_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "majority_class": majority_class,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "valid_predictions_pct": valid_predictions / len(y_true) * 100,
        "class_metrics": {k: v for k, v in class_report.items() if k in risk_levels or k in ["macro avg", "weighted avg"]},
        "confusion_matrix": conf_matrix.tolist(),
        "confusion_matrix_labels": risk_levels + ["Unknown"],
        "predictions": {"true": y_true, "pred": y_pred},
        "per_sample_accuracy": accuracy_scores,
        "mean_sample_accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0,
        "sureness_scores": sureness_scores,
        "mean_sureness": sum(sureness_scores) / len(sureness_scores) if sureness_scores else 0,
        "sureness_by_class": {}, 
    }
    
    # Print detailed results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f} (Baseline: {baseline_accuracy:.4f}, Difference: {accuracy - baseline_accuracy:.4f})")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print("\nPer-Class Metrics:")
    
    for risk_level in risk_levels:
        if risk_level in class_report:
            metrics = class_report[risk_level]
            print(f"  {risk_level}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    
    # Calculate sureness by class
    for risk_level in risk_levels:
        level_sureness = []
        for i, true_label in enumerate(y_true):
            if true_label == risk_level:
                level_sureness.append(sureness_scores[i])
        
        if level_sureness:
            detailed_metrics["sureness_by_class"][risk_level] = {
                "mean": sum(level_sureness) / len(level_sureness),
                "min": min(level_sureness),
                "max": max(level_sureness),
                "std": np.std(level_sureness)
            }

    # Save evaluation results
    if results_file:
        with open(results_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        print(f"\nEvaluation results saved to {results_file}")
    
    return detailed_metrics

def train_forest_fire_model(train_dataset, eval_dataset, ckpt=False):
    """Train a model for forest fire risk assessment using the provided datasets"""
    # Ensure the output directory exists
    os.makedirs("./outputs-forestfire", exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Display a few samples to verify
    display_dataset_sample(train_dataset)
    print(f"Dataset split: {len(train_dataset)} training and {len(eval_dataset)} validation samples")

    # Configure training
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_train_epochs=epochs,
        save_total_limit=save_total_limit,
        save_steps=save_steps,
        max_grad_norm=0.1,
        temperature=training_temperature,
        report_to="none",
        output_dir="./outputs-forestfire",
    )

    print("Setting up trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            format_reward_func,
            risk_level_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Initialize and add accuracy tracker
    accuracy_tracker = AccuracyTracker(window_size=20)
    trainer.add_callback(AccuracyCallback(accuracy_tracker))

    # Train the model
    print("Starting training...")
    if ckpt:
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
        trainer_stats = trainer.train()

    # Save the model
    print("Training complete! Saving model...")
    model.save_pretrained("forest-fire-reasoning-model")
    print("Model saved to 'forest-fire-reasoning-model'")
    
    return model, tokenizer, trainer_stats

def calculate_dataset_stats(dataset):
    """Calculate risk level distribution in the dataset"""
    risk_counts = {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0, "Unknown": 0}
    
    for item in dataset:
        answer = item.get('answer', '')
        if answer in risk_counts:
            risk_counts[answer] += 1
        else:
            risk_counts["Unknown"] += 1
    
    total = sum(risk_counts.values())
    risk_distribution = {k: v/total for k, v in risk_counts.items() if k != "Unknown"}
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total samples: {total}")
    
    # Print counts and percentages
    for risk, count in risk_counts.items():
        if risk != "Unknown" or risk_counts["Unknown"] > 0:
            percentage = count/total * 100
            print(f"{risk}: {count} samples ({percentage:.1f}%)")
    
    # Baseline metrics
    majority_class = max(risk_distribution, key=risk_distribution.get)
    baseline_accuracy = risk_distribution[majority_class]
    
    print("\nMajority class: " + majority_class)
    print(f"Baseline accuracy (always predicting majority): {baseline_accuracy*100:.1f}%")
    print(f"Random baseline (3 classes): 33.3%")
    print("="*50)
    
    return risk_distribution

def evaluate_model_comparison(test_dataset, output_dir="./outputs-forestfire", checkpoint_dir=None):
    """Compare base model performance vs fine-tuned model
    
    Args:
        test_dataset: Dataset to evaluate on
        output_dir: Directory for outputs
        checkpoint_dir: Optional path to a checkpoint directory to evaluate
    """
    import time
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define display names for models (used for printing and file names)
    base_display_name = "Base Model"
    
    if checkpoint_dir:
        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
        finetuned_display_name = f"Fine-tuned Model (Checkpoint {checkpoint_name})"
    else:
        finetuned_display_name = "Fine-tuned Model"
    
    # Define models to evaluate
    models_to_evaluate = [
        {
            "name": base_display_name,
            "load_func": lambda: load_base_model(model_name)
        },
        {
            "name": finetuned_display_name,
            "load_func": lambda: load_finetuned_model(model_name, "forest-fire-reasoning-model", checkpoint_dir)
        }
    ]
    
    # Initialize results dictionary
    comparison_results = {}
    
    for model_info in models_to_evaluate:
        display_name = model_info["name"]
        print(f"\n{'='*50}")
        print(f"EVALUATING {display_name}")
        print(f"{'='*50}")
        
        try:
            # Load model
            print(f"Loading {display_name}...")
            start_time = time.time()
            model, tokenizer = model_info["load_func"]()
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f} seconds")
            
            # Evaluate the model
            file_safe_name = display_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            results = evaluate_model(model, tokenizer, test_dataset, 
                                     f"{output_dir}/{file_safe_name}_results.json")
            
            # Store results for comparison
            comparison_results[display_name] = results
            
        except Exception as e:
            print(f"Error evaluating {display_name}: {str(e)}")
            comparison_results[display_name] = {"error": str(e)}
    
    # Create comparison visualizations
    if len(comparison_results) > 1 and all("error" not in comparison_results[model] for model in comparison_results):
        create_comparison_visualizations(comparison_results, output_dir)
    
    # Save overall comparison results
    with open(os.path.join(output_dir, 'model_comparison.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nComparison results saved to {os.path.join(output_dir, 'model_comparison.json')}")
    
    return comparison_results

def load_base_model(model_name):
    """Load the base model without fine-tuning"""
    try:
        # First try with unsloth if available
        print("Using Unsloth for base model loading")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
        )
        # Important: Prepare model for inference
        model = FastLanguageModel.for_inference(model)
    except (ImportError, Exception) as e:
        # Fall back to regular transformers loading
        print(f"Falling back to regular transformers loading: {str(e)}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def load_finetuned_model(base_model_name, adapter_path, checkpoint_dir=None):
    """Load the fine-tuned model with adapter
    
    Args:
        base_model_name: Name of the base model
        adapter_path: Path to the saved adapter/fine-tuned model
        checkpoint_dir: Path to a checkpoint directory, if using checkpoint instead of final model
    """
    using_unsloth = False
    try:
        # First try with unsloth for base model loading if available
        from unsloth import FastLanguageModel
        print("Using Unsloth for base model loading")
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
        )
        using_unsloth = True
    except (ImportError, Exception) as e:
        # Fall back to regular transformers loading
        print(f"Falling back to regular transformers loading: {str(e)}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load the adapter path
    actual_adapter_path = checkpoint_dir if checkpoint_dir else adapter_path
    print(f"Loading adapter from: {actual_adapter_path}")
    
    try:
        model = PeftModel.from_pretrained(base_model, actual_adapter_path)
        
        # If we're using Unsloth, prepare model for inference
        if using_unsloth:
            from unsloth import FastLanguageModel
            model = FastLanguageModel.for_inference(model)
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading adapter: {str(e)}")
        raise

def create_comparison_visualizations(comparison_results, output_dir):
    """Create visualizations comparing model performance"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    
    # Extract model names
    model_names = list(comparison_results.keys())
    
    # Check for errors
    for model_name in model_names:
        if "error" in comparison_results[model_name]:
            print(f"Skipping visualizations for {model_name} due to errors")
            return
    
    # 1. Overall metrics comparison
    metrics = ["accuracy", "f1_weighted", "f1_macro"]
    metric_values = {model: [comparison_results[model][m] for m in metrics] for model in model_names}
    
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    x = np.arange(len(metrics))
    width = 0.35
    
    # Plot bars for each model
    for i, (model, values) in enumerate(metric_values.items()):
        ax.bar(x + (i - len(model_names)/2 + 0.5) * width, values, width, label=model)
    
    # Add baseline accuracy
    baseline = comparison_results[model_names[0]]["baseline_accuracy"]
    ax.axhline(y=baseline, linestyle='--', color='r', alpha=0.7, label=f'Majority Baseline ({baseline:.3f})')
    
    # Add random baseline (0.33 for 3 classes)
    ax.axhline(y=0.33, linestyle=':', color='gray', alpha=0.7, label='Random Baseline (0.333)')
    
    # Customize plot
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/visualizations/overall_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class F1 scores
    risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
    
    # Extract per-class F1 scores
    class_f1 = {}
    for model in model_names:
        try:
            class_f1[model] = [comparison_results[model]["class_metrics"][level]["f1-score"] for level in risk_levels]
        except KeyError:
            print(f"Warning: Missing class metrics for {model}. Skipping per-class visualization.")
            return
    
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    x = np.arange(len(risk_levels))
    
    # Plot bars for each model
    for i, (model, values) in enumerate(class_f1.items()):
        ax.bar(x + (i - len(model_names)/2 + 0.5) * width, values, width, label=model)
    
    # Customize plot
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(risk_levels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/visualizations/per_class_f1_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion matrices for each model
    for model in model_names:
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
            
            plt.savefig(f"{output_dir}/visualizations/confusion_matrix_{safe_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
        except (KeyError, Exception) as e:
            print(f"Warning: Error creating confusion matrix for {model}: {str(e)}")
    
    # 4. Improvement over baseline
    try:
        improvement = {
            model: comparison_results[model]["accuracy"] - comparison_results[model]["baseline_accuracy"]
            for model in model_names
        }
        
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        
        # Determine colors based on number of models
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'] 
        colors = colors[:len(model_names)]
        
        ax.bar(model_names, [improvement[model] for model in model_names], color=colors)
        
        # Add zero line
        ax.axhline(y=0, linestyle='-', color='k', alpha=0.3)
        
        # Customize plot
        ax.set_ylabel('Improvement over Majority Baseline')
        ax.set_title('Model Improvement over Baseline')
        plt.xticks(rotation=15, ha='right')
        
        # Add exact values above bars
        for i, model in enumerate(model_names):
            value = improvement[model]
            ax.text(i, value + 0.01, f"{value:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/visualizations/improvement_over_baseline.png", dpi=300, bbox_inches='tight')
        plt.close()
    except (KeyError, Exception) as e:
        print(f"Warning: Error creating improvement visualization: {str(e)}")
    
    print(f"Comparison visualizations saved to {output_dir}/visualizations/")

if __name__ == "__main__":
    # Prepare dataset
    print("Preparing dataset...")
    full_dataset = prepare_forest_fire_dataset(max_samples=train_samples)
    
    if full_dataset is None or len(full_dataset) == 0:
        print("Failed to load dataset or dataset is empty. Aborting.")
        exit(1)
    
    # Check if we already have saved splits
    if os.path.exists(os.path.join(data_dir, "train_dataset.json")) and \
       os.path.exists(os.path.join(data_dir, "test_dataset.json")):
        print("Found existing dataset splits. Loading them...")
        train_dataset, test_dataset = load_dataset_splits(data_dir)
    else:
        print("Creating and saving new dataset splits...")
        train_dataset, test_dataset = split_and_save_dataset(full_dataset, val_ratio, data_dir)
    
    # Calculate and display dataset statistics
    print("\nAnalyzing training dataset statistics...")
    train_risk_dist = calculate_dataset_stats(train_dataset)
    
    print("\nAnalyzing test dataset statistics...")
    test_risk_dist = calculate_dataset_stats(test_dataset)
    
    # Create accuracy tracker with actual risk distribution
    accuracy_tracker = AccuracyTracker(window_size=20, risk_distribution=train_risk_dist)
    
    # Ask user which mode to run
    mode = input("\nChoose mode (train, eval, both, compare) [both]: ").strip().lower() or "both"
    
    if mode == "compare":
        # Load existing dataset splits
        if os.path.exists(os.path.join(data_dir, "test_dataset.json")):
            print("Loading test dataset for comparison...")
            _, test_dataset = load_dataset_splits(data_dir)
            
            # Check if user wants to use a checkpoint
            checkpoint_option = input("\nEvaluate from a checkpoint? (y/n) [n]: ").strip().lower() or "n"
            checkpoint_dir = None
            
            if checkpoint_option.startswith("y"):
                # Get checkpoint directory
                output_dir = "./outputs-forestfire"
                checkpoints = []
                
                # List available checkpoints
                if os.path.exists(output_dir):
                    for item in os.listdir(output_dir):
                        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, item)):
                            checkpoints.append(item)
                
                if checkpoints:
                    print("\nAvailable checkpoints:")
                    for i, checkpoint in enumerate(sorted(checkpoints)):
                        print(f"{i+1}. {checkpoint}")
                    
                    # Ask user to choose a checkpoint
                    choice = input("\nEnter checkpoint number (or press Enter for latest): ").strip()
                    if choice and choice.isdigit() and 0 < int(choice) <= len(checkpoints):
                        checkpoint_dir = os.path.join(output_dir, sorted(checkpoints)[int(choice)-1])
                    elif not choice:  # Default to latest checkpoint
                        checkpoint_dir = os.path.join(output_dir, sorted(checkpoints)[-1])
                    else:
                        print("Invalid selection. Using the latest checkpoint.")
                        checkpoint_dir = os.path.join(output_dir, sorted(checkpoints)[-1])
                else:
                    print("No checkpoints found in ./outputs-forestfire directory.")
            
            # Run comparative evaluation
            print("\nStarting model comparison...")
            comparison_results = evaluate_model_comparison(test_dataset, checkpoint_dir=checkpoint_dir)
        else:
            print("Test dataset not found. Please create dataset splits first.")

    elif mode in ["train", "both"]:
        # Train the model
        model, tokenizer, stats = train_forest_fire_model(train_dataset, test_dataset, resume_training)
        
        if mode == "both":
            # Evaluate the model on test set
            print("Evaluating model on test set...")
            eval_results = evaluate_model(model, tokenizer, test_dataset)
            
    elif mode == "eval":
        # Load an existing model for evaluation
        print("Loading existing model for evaluation...")
        try:
            # Check if user wants to use a checkpoint
            checkpoint_option = input("\nEvaluate from a checkpoint? (y/n) [n]: ").strip().lower() or "n"
            checkpoint_dir = None
            
            if checkpoint_option.startswith("y"):
                # Get checkpoint directory
                output_dir = "./outputs-forestfire"
                checkpoints = []
                
                # List available checkpoints
                if os.path.exists(output_dir):
                    for item in os.listdir(output_dir):
                        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, item)):
                            checkpoints.append(item)
                
                if checkpoints:
                    print("\nAvailable checkpoints:")
                    for i, checkpoint in enumerate(sorted(checkpoints)):
                        print(f"{i+1}. {checkpoint}")
                    
                    # Ask user to choose a checkpoint
                    choice = input("\nEnter checkpoint number (or press Enter for latest): ").strip()
                    if choice and choice.isdigit() and 0 < int(choice) <= len(checkpoints):
                        checkpoint_dir = os.path.join(output_dir, sorted(checkpoints)[int(choice)-1])
                    elif not choice:  # Default to latest checkpoint
                        checkpoint_dir = os.path.join(output_dir, sorted(checkpoints)[-1])
                    else:
                        print("Invalid selection. Using the latest checkpoint.")
                        checkpoint_dir = os.path.join(output_dir, sorted(checkpoints)[-1])
                    
                    # Use the checkpoint
                    adapter_path = checkpoint_dir
                    print(f"Using checkpoint: {adapter_path}")
                else:
                    print("No checkpoints found in ./outputs-forestfire directory.")
                    adapter_path = "forest-fire-reasoning-model"
            else:
                adapter_path = "forest-fire-reasoning-model"
            
            # Load model from adapter
            model, tokenizer = load_finetuned_model(model_name, adapter_path, checkpoint_dir if checkpoint_option.startswith("y") else None)
            
            # Evaluate the model
            print("Evaluating model on test set...")
            eval_results = evaluate_model(model, tokenizer, test_dataset, 
                                        os.path.join("./outputs-forestfire", "eval_results.json"))
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please train a model first or check that the model path exists.")
    
    print("Process completed!")