import requests
import json
import random
import re
import time
from typing import Dict, List, Union, Optional
import os

class ForestFireDataGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate"):
        self.ollama_url = ollama_url
        self.headers = {"Content-Type": "application/json"}
        
    def generate_sensor_data(self) -> Dict[str, str]:
        """Generate synthetic sensor data with realistic correlations"""
        # Base temperature (30-45°C)
        temp = random.uniform(20, 45)
        
        # Humidity inversely correlates with temperature
        humidity_max = 60 - (temp - 30)
        humidity = max(10, random.uniform(10, humidity_max))
        
        # Wind affects temperature and humidity slightly
        wind_speed = random.uniform(5, 25)
        temp += wind_speed * 0.1
        humidity -= wind_speed * 0.2
        
        # NDVI (Vegetation health decreases in high temp/low humidity)
        ndvi_base = random.uniform(0.1, 0.9)
        ndvi = max(0.2, ndvi_base - (temp/100) + (humidity/200))
        
        # Smoke detection more likely with high temp/low humidity
        smoke_base = random.uniform(0.0, 0.3)
        smoke = min(1.0, smoke_base + (temp/100) - (humidity/200))
        
        # Rainfall affects humidity and NDVI
        rainfall = 0 if random.random() < 0.7 else random.uniform(0, 10)
        if rainfall > 0:
            humidity += rainfall * 2
            ndvi += rainfall * 0.05
        
        # Drought index affected by temp, humidity, rainfall
        drought_base = random.uniform(0.2, 0.8)
        drought = min(1.0, drought_base + (temp/100) - (humidity/200) - (rainfall/20))
        
        return {
            "Temperature": f"{temp:.1f}°C",
            "Humidity": f"{min(100, max(0, humidity)):.1f}%",
            "Wind Speed": f"{wind_speed:.1f} km/h",
            "NDVI (Vegetation Health)": f"{min(1.0, max(0, ndvi)):.2f}",
            "Smoke Detection Index": f"{min(1.0, max(0, smoke)):.2f}",
            "Rainfall in the last 24 hours": f"{rainfall:.1f} mm",
            "Drought Index": f"{min(1.0, max(0, drought)):.2f}"
        }

    def build_prompt(self, sensor_data: Dict[str, str]) -> str:
        """Build a detailed prompt with the sensor data"""
        sensor_text = "\n".join([f"- {key}: {value}" for key, value in sensor_data.items()])
        
        prompt = f"""As a forest fire detection analyst, assess the fire risk using this sensor data. Guidelines:

- Temperature >40°C: high risk (more is higher risk)
- Humidity <20%: high risk (less is higher risk)
- Wind >20 km/h: increased risk (more is higher risk)
- NDVI <0.3: high risk (dry vegetation, less is higher risk)
- Smoke Index >0.6: possible fire (more is higher risk)
- Recent rainfall reduces risk
- Drought Index >0.7: high risk (more is higher risk)

Respond ONLY in this format:

<answer>
[ONLY state: Low Risk, Moderate Risk, or High Risk]
</answer>

Sensor readings:

{sensor_text}

What is the forest fire risk level?"""
        return prompt

    def query_ollama(self, prompt: str, retries: int = 3) -> Optional[str]:
        """Query Ollama API with proper streaming response handling"""
        payload = {
            "model": "deepseek-r1:32b", # "qwen2.5:14b",
            "prompt": prompt,
            "stream": False,  # Disable streaming for simpler response handling
            "max_tokens": 2048,
            "temperature": 0.7
        }
        
        for attempt in range(retries):
            try:
                print(f"Attempt {attempt + 1}: Querying Ollama...")
                response = requests.post(
                    self.ollama_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                # Handle the response
                full_response = ""
                for line in response.text.strip().split('\n'):
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            full_response += data["response"]
                    except json.JSONDecodeError:
                        continue
                
                if full_response:
                    print("Successfully received response from Ollama")
                    return full_response.strip()
                
                print("No valid response content found")
                
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {str(e)}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("All retries failed")
        
        return None
    
    def extract_final_answer(self, response: str) -> str:
        """Extract the final answer with improved parsing"""
        if not response:
            return "Error: No response"
            
        print("Raw response:", response + "...")
            
        # Try to extract content between answer tags
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            
            # Validate the answer matches expected format
            risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
            for level in risk_levels:
                if level.lower() in answer.lower():
                    return level
            
            print(f"Invalid risk level found in answer: {answer}")
            return "Error: Invalid risk level"
        
        print("No answer tags found in response")
        return "Error: No answer tags found"
    
    def extract_reasoning_and_answer(self, response: str) -> dict:
        """Extract both reasoning (from think tags) and final answer from response
        
        Args:
            response: The full response from the LLM
            
        Returns:
            dict: A dictionary with 'reasoning' and 'answer' keys
        """
        result = {
            'reasoning': '',
            'answer': 'Error: No response'
        }
        
        if not response:
            return result
                
        # Extract reasoning from think tags
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", response, re.DOTALL)
        if think_match:
            result['reasoning'] = think_match.group(1).strip()
        
        # Extract answer from answer tags
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            
            # Validate the answer matches expected format
            risk_levels = ["Low Risk", "Moderate Risk", "High Risk"]
            for level in risk_levels:
                if level.lower() in answer_text.lower():
                    result['answer'] = level
                    return result
            
            result['answer'] = "Error: Invalid risk level"
        else:
            result['answer'] = "Error: No answer tags found"
        
        return result

    def validate_dataset_entry(self, entry: Dict) -> bool:
        """Validate a dataset entry has all required fields"""
        required_fields = ["prompt", "answer", "metadata"]
        if not all(field in entry for field in required_fields):
            return False
            
        if not isinstance(entry["metadata"], dict) or \
           "sensor_data" not in entry["metadata"] or \
           "full_response" not in entry["metadata"]:
            return False
            
        valid_answers = ["Low Risk", "Moderate Risk", "High Risk", 
                        "Error: No response", "Error: Invalid risk level", 
                        "Error: No answer tags found"]
        if entry["answer"] not in valid_answers:
            return False
            
        return True

def generate_dataset(num_samples: int = 10, 
                    output_file: str = "forest_fire_dataset.json",
                    append: bool = True) -> None:
    """Generate the forest fire risk assessment dataset
    
    Args:
        num_samples: Number of new samples to generate
        output_file: Path to the output JSON file
        append: If True, append to existing file; if False, create new file
    """
    generator = ForestFireDataGenerator()
    dataset = []
    error_count = 0
    
    # Load existing dataset if append=True and file exists
    if append and os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                dataset = json.load(f)
            print(f"Loaded existing dataset with {len(dataset)} samples")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading existing dataset: {str(e)}")
            print("Creating new dataset instead")
            dataset = []
    
    current_count = len(dataset)
    total_samples = current_count + num_samples
    
    print(f"\nStarting dataset generation for {num_samples} new samples...")
    print(f"Currently have {current_count} samples, aiming for {total_samples} total")
    print("=" * 50)
    
    for i in range(num_samples):
        try:
            print(f"\nGenerating sample {i+1}/{num_samples} (will be #{current_count+i+1} overall)")
            print("-" * 30)
            
            # Generate sensor data
            sensor_data = generator.generate_sensor_data()
            print("Sensor data generated")
            
            # Build prompt
            prompt = generator.build_prompt(sensor_data)
            print("Prompt built")
            
            # Get LLM response
            response = generator.query_ollama(prompt)
            
            if response is None:
                print("Failed to get response from Ollama")
                error_count += 1
                continue
                
            # Extract reasoning and answer
            extracted = generator.extract_reasoning_and_answer(response)
            final_answer = extracted['answer']
            reasoning = extracted['reasoning']
            
            print(f"Extracted answer: {final_answer}")
            print(f"Extracted reasoning length: {len(reasoning)} characters")
            
            # Create dataset entry
            entry = {
                "prompt": prompt,
                "answer": final_answer,
                "metadata": {
                    "sensor_data": sensor_data,
                    "full_response": response,
                    "reasoning": reasoning  # Add reasoning to metadata
                }
            }
            
            # Modified validation to check for reasoning
            if final_answer.startswith("Error:"):
                print(f"Invalid response: {final_answer}")
                error_count += 1
                continue
            
            # Add to dataset
            dataset.append(entry)
            print(f"Sample {i+1} generated successfully")
            
            # Brief pause between samples
            time.sleep(1)
            
            # Save after each successful addition to avoid losing progress
            with open(output_file, "w") as f:
                json.dump(dataset, f, indent=2)
            
        except Exception as e:
            print(f"Error generating sample {i+1}: {str(e)}")
            error_count += 1
    
    # Final save of dataset
    if dataset:
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)
        
        print("\nDataset generation complete:")
        print(f"- Started with: {current_count} samples")
        print(f"- Added: {len(dataset) - current_count} new samples")
        print(f"- Failed attempts: {error_count}")
        print(f"- Total dataset size: {len(dataset)} samples")
        print(f"- Dataset saved to: {output_file}")
    else:
        print("\nError: No valid samples were generated")

if __name__ == "__main__":
    generate_dataset(num_samples=1000, append=True)