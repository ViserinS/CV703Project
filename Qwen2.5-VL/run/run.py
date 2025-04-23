import os
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
from tqdm import tqdm

def load_model_and_processor():
    """Load the Qwen2.5-VL model and processor."""
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/output/cassava_disease_deepseek_v3_1epoch", 
        torch_dtype=torch.bfloat16, 
        device_map={'': 3}
    )
    
    processor = AutoProcessor.from_pretrained(
        "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/output/cassava_disease_deepseek_v3_1epoch"
    )
    
    return model, processor

def diagnose_image(model, processor, image_path, prompt):
    """Generate diagnosis for a given cassava plant image."""
    try:
        image = Image.open(image_path).convert('RGB')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        
        # 去掉 input_ids 部分，只保留新生成的 tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def extract_diagnosis(text):
    """Extract the disease diagnosis from the model's output text."""
    text_lower = text.lower()
    cbb_patterns = [r"cassava bacterial blight\b", r"\bcbb\b", r"bacterial blight", r"xanthomonas"]
    cmd_patterns = [r"cassava mosaic disease\b", r"\bcmd\b", r"mosaic disease", r"mosaic virus"]
    healthy_patterns = [r"healthy", r"not diseased", r"no disease", r"no sign of disease"]
    
    for p in cbb_patterns:
        if re.search(p, text_lower):
            return "CBB"
    for p in cmd_patterns:
        if re.search(p, text_lower):
            return "CMD"
    for p in healthy_patterns:
        if re.search(p, text_lower):
            return "Healthy"
    return "Other"

def main():
    image_dir = "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/course_project_dataset"
    output_file = "cassava_diagnosis_results_deepseek_v3_1epoch.json"
    prompt = "What disease is affecting this cassava plant? Please analyze the symptoms visible in the image and provide your diagnosis with reasoning"
    
    # 如果已经存在旧的结果文件，先清空
    if os.path.exists(output_file):
        open(output_file, "w").close()
    
    model, processor = load_model_and_processor()
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    results = []
    
    for filename in tqdm(image_files):
        image_path = os.path.join(image_dir, filename)
        output_text = diagnose_image(model, processor, image_path, prompt)
        if output_text:
            pred = extract_diagnosis(output_text)
            record = {
                "image_path": image_path,
                "prediction": pred,
                "full_output": output_text
            }
            results.append(record)
            
            # 每处理一个样本就写入一次文件
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            
            print(f"Processed {filename}: Prediction: {pred}")
    
    # 最终打印统计
    print("\nDiagnosis Summary:")
    diagnoses = [r["prediction"] for r in results]
    for diag in ["CBB", "CMD", "Healthy", "Other"]:
        cnt = diagnoses.count(diag)
        print(f"{diag}: {cnt} images ({cnt/len(diagnoses)*100:.1f}%)")
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()