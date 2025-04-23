import json
import os
import argparse
import random
import time
import requests
from PIL import Image
from io import BytesIO
import base64
from tqdm import tqdm

# Add a list of diverse questions
QUESTION_FORMATS = [
    "What disease is affecting this cassava plant? Please analyze the symptoms visible in the image and provide your diagnosis with reasoning.",
    "Can you identify the disease on this cassava plant? Please provide a detailed analysis of the symptoms and your final diagnosis.",
    "This cassava plant appears diseased. What condition is it suffering from? Analyze the visible symptoms and explain your reasoning.",
    "Please diagnose the disease affecting this cassava plant. What symptoms do you observe, and what is your conclusion?",
    "What disease has infected this cassava? Provide a thorough analysis of the symptoms visible in the image and your final diagnosis.",
    "Identify the disease affecting this cassava plant. Please analyze the visual symptoms and explain how you reached your diagnosis.",
    "This is a cassava plant showing disease symptoms. Can you diagnose the condition based on the visible symptoms?",
    "What pathogen is affecting this cassava plant? Analyze the visible symptoms in detail and provide your diagnosis.",
    "Please examine this cassava plant and identify the disease. Describe the symptoms you observe and provide your diagnosis with reasoning.",
    "What is wrong with this cassava plant? Analyze the symptoms and identify the specific disease affecting it."
]

def encode_image_to_base64(image_path):
    """Encode image to base64 for API request"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_claude_response(api_key, model, image_path, disease_type, question=None, system_prompt=None):
    """
    Get diagnosis from Claude model using the Anthropic API
    
    Args:
        api_key: Anthropic API key
        model: Model to use (e.g., "claude-3-7-sonnet-20250219")
        image_path: Path to the image file
        disease_type: Either "cbb" for Cassava Bacterial Blight or "cmd" for Cassava Mosaic Disease
        question: Custom question to ask (if None, a random one will be selected)
        system_prompt: Custom system prompt (optional)
    
    Returns:
        Claude's response text and the question used
    """
    # Encode image
    base64_image = encode_image_to_base64(image_path)
    
    # Select a random question if none provided
    if question is None:
        question = random.choice(QUESTION_FORMATS)
    
    # Default system prompt with instructions
    if system_prompt is None:
        if disease_type == "cbb":
            disease_name = "Cassava Bacterial Blight (CBB)"
            disease_desc = ("a bacterial disease showing dark brown angular spots, yellowish margins, "
                           "brown-black areas along leaf veins, and dry papery texture in affected areas")
        else:  # cmd
            disease_name = "Cassava Mosaic Disease (CMD)"
            disease_desc = ("a viral disease showing irregular light/dark green mosaic patterns, "
                           "yellow-green mottled areas, leaf curling, and deformation")
        
        system_prompt = f"""You are an expert in plant pathology specializing in cassava diseases.
Analyze the image of a cassava plant and identify {disease_name}.

IMPORTANT INSTRUCTIONS:
1. The plant in the image has {disease_name} ({disease_desc}).
2. Provide a detailed chain-of-thought analysis of visual symptoms visible in the image.
3. Focus on SPECIFIC, CONCRETE visual features (colors, textures, patterns, shapes).
4. Use precise color descriptions (e.g., "dark brown spots" not just "spots").
5. Describe texture details (e.g., "dry papery texture," "bumpy surface").
6. DO NOT use abstract technical terms like "water-soaked lesions" - instead describe the specific appearance.
7. Describe what you actually see in terms of colors, patterns and textures.
8. Conclude with a definitive diagnosis of {disease_name}.

Format your response in two paragraphs:
Paragraph 1: Detailed symptom analysis with reasoning based on visual features.
Paragraph 2: Final diagnosis statement identifying the disease with scientific name.
"""

    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    # Prepare messages for Claude API
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        }
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "system": system_prompt,
        "max_tokens": 800
    }
    
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        return result["content"][0]["text"], question
    except Exception as e:
        print(f"Error getting Claude response: {e}")
        if 'response' in locals():
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
        return None, question

def create_dataset_with_claude(image_dir, output_file, api_key, model="claude-3-7-sonnet-20250219", relative_paths=False, sample_size=None, custom_prompt=None, fixed_question=None):
    """
    Create a dataset using Claude models
    
    Args:
        image_dir: Directory containing the images
        output_file: Path to save the JSON dataset
        api_key: Anthropic API key
        model: Model to use (default: "claude-3-7-sonnet-20250219")
        relative_paths: If True, use relative paths in the dataset
        sample_size: Number of images to sample (None for all)
        custom_prompt: Custom system prompt (optional)
        fixed_question: If provided, use this question for all images instead of random ones
    """
    print(f"Creating dataset using {model}...")
    
    # List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sample a subset if requested
    if sample_size is not None and sample_size < len(image_files):
        print(f"Sampling {sample_size} images from {len(image_files)}")
        image_files = random.sample(image_files, sample_size)
    
    dataset = []
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        # Determine the image path
        if relative_paths:
            image_path_for_json = image_file  # Just the filename for relative paths
        else:
            image_path_for_json = os.path.join(image_dir, image_file)  # Full path
        
        full_image_path = os.path.join(image_dir, image_file)
        
        # Determine disease type from filename
        if "cbb" in image_file.lower():
            disease_type = "cbb"
        elif "cmd" in image_file.lower():
            disease_type = "cmd"
        else:
            print(f"Warning: Could not determine disease type for {image_file}, skipping")
            continue
        
        # Get Claude response - use fixed_question if provided
        claude_response, used_question = get_claude_response(
            api_key, 
            model, 
            full_image_path, 
            disease_type, 
            question=fixed_question, 
            system_prompt=custom_prompt
        )
        
        if claude_response:
            # Create dataset entry
            entry = {
                "image": image_path_for_json,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{used_question}"
                    },
                    {
                        "from": "assistant",
                        "value": claude_response
                    }
                ]
            }
            
            dataset.append(entry)
            
            # Sleep to avoid hitting API rate limits
            time.sleep(0.5)
        else:
            print(f"Warning: Failed to get Claude response for {image_file}, skipping")
    
    # Write dataset to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Generated dataset with {len(dataset)} entries, saved to {output_file}")
    
    # Print distribution of disease types
    cbb_count = sum(1 for entry in dataset if "cbb" in entry["image"].lower())
    cmd_count = sum(1 for entry in dataset if "cmd" in entry["image"].lower())
    print(f"Disease distribution: CBB: {cbb_count}, CMD: {cmd_count}")
    
    return dataset

def split_dataset(dataset, train_ratio=0.8, output_dir="."):
    """Split the dataset into training and validation sets"""
    random.shuffle(dataset)  # Shuffle to ensure random distribution
    
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    # Save the splits
    train_file = os.path.join(output_dir, "cassava_train.json")
    val_file = os.path.join(output_dir, "cassava_val.json")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"Split dataset: {len(train_data)} training examples, {len(val_data)} validation examples")
    print(f"Training data saved to: {train_file}")
    print(f"Validation data saved to: {val_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Cassava disease dataset using Claude 3.7 API")
    parser.add_argument("--image_dir", required=True, help="Directory containing the images")
    parser.add_argument("--output_file", default="cassava_disease_dataset_claude.json", help="Path to save the JSON dataset")
    parser.add_argument("--api_key", required=True, help="Anthropic API key")
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219", help="Claude model to use")
    parser.add_argument("--relative_paths", action="store_true", help="Use relative paths in dataset")
    parser.add_argument("--sample_size", type=int, help="Number of images to sample (optional)")
    parser.add_argument("--split", action="store_true", help="Split dataset into train/val")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio for training set if splitting")
    parser.add_argument("--output_dir", default=".", help="Directory to save split datasets")
    parser.add_argument("--fixed_question", help="Use a fixed question for all images instead of random ones")
    parser.add_argument("--vary_questions", action="store_true", help="Use varied questions for each image")
    parser.add_argument("--sample_limit", type=int, default=500, help="Limit the number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Determine whether to use fixed or varied questions
    fixed_q = None if args.vary_questions else (args.fixed_question or QUESTION_FORMATS[0])
    
    # Create dataset
    dataset = create_dataset_with_claude(
        args.image_dir, 
        args.output_file, 
        args.api_key,
        args.model,
        args.relative_paths,
        args.sample_size if args.sample_size else args.sample_limit,
        fixed_question=fixed_q
    )
    
    # Split if requested
    if args.split and dataset:
        split_dataset(dataset, args.train_ratio, args.output_dir)
    
    # Print a sample entry
    if dataset:
        print("\nSample entry:")
        print(json.dumps(dataset[0], indent=2, ensure_ascii=False))
        