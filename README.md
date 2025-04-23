## Introduction

This is our CV703 project, Topic: Reasoning Agro-GPT: Bridging Chain-of-Thought Gaps



## Installation

#### Install environment

please run the following command to install the environment

```bash
conda create -n qwen python=3.12
conda activate qwen
pip install -r requirements.txt
sudo apt install git-lfs
git lfs install
```

[option] for more detail you can visit this website [Qwen](https://github.com/QwenLM/Qwen2.5-VL)



#### Get baseline model

you should download the [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), you can run the following command:

```bash
cd Qwen2.5-VL/qwen-vl-finetune/
git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
cd Qwen2.5-VL-3B-Instruct/
git lfs pull
```
You can access fine-tuned model through this link [GoogleDrive](https://drive.google.com/file/d/1eq5q5TQbNgqy3Z01Pjp3Vhja6MZWJed4/view?usp=sharing)


## Running

#### Data Generation

If you want to regenerate the CoT data, please run the following command and make sure you have an API key.

```bash
python Qwen2.5-VL/qwen-vl-finetune/dataset/data_g.py --image_dir xx --output_file xx --api_key xx --model xx --vary_questions --sample_limit 1000
```

we use the following prompt to generate data

```python
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
```

And those files are generated data:

- Qwen2.5-VL/qwen-vl-finetune/dataset/cassava_disease_dataset_claude.json
- Qwen2.5-VL/qwen-vl-finetune/dataset/cassava_disease_dataset_deepseek_R1.json
- Qwen2.5-VL/qwen-vl-finetune/dataset/cassava_disease_dataset_deepseek_V3.json

This folder has image of the dataset:

- Qwen2.5-VL/qwen-vl-finetune/dataset/course_project_dataset



#### Evaluation CoT data

we use the code from [Llava-o1](https://github.com/mbzuai-oryx/LlamaV-o1/blob/main/eval/get_result.py)

please set API first and path first, and run:

```
python Qwen2.5-VL/qwen-vl-finetune/dataset/get_result.py
```





#### Model Fine-tuning

And run the following command to fine-tuning Qwen model:

```
sh Qwen2.5-VL/qwen-vl-finetune/scripts/sft_ag_3b.sh
```

you should Change the model path to the full path first

Since we set:

- NPROC_PER_NODE=6
- batch_size=4
- grad_accum_steps=4



So,

$$ \text{Effective Batch Size} = 4 \times 4 \times 6 = 96$$



if you want to use different datasets, you can change the `dataset` in `sft_ag_3b.sh`

now you have three options:

- cassava_disease_claude
- cassava_disease_deepseek_R1

- cassava_disease_deepseek_v3

which defined in `Qwen2.5-VL/qwen-vl-finetune/qwenvl/data/__init__.py`



#### Reasoning

Please run the following command to reason:

```bash
python Qwen2.5-VL/run/run.py
```

You need to set model path, `output_file`, `image_dir`, and GPU first

The output will be stored as json file

#### Evaluation

And then run the following command to evaluate the model:

```
python Qwen2.5-VL/run/acc.py
```

You should set the path of generated json file in Reasoning stage

And you will get the accuracy.




