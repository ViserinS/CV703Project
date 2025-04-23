import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

CASSAVA_DISEASE_TRAIN = {
    "annotation_path": "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/annotation/train_set.json",
    "data_path": "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/course_project_dataset",
}

CASSAVA_DISEASE_TEST = {
    "annotation_path": "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/annotation/test_set.json",
    "data_path": "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/course_project_dataset",
}

CASSAVA_DISEASE_CLAUDE = {
    "annotation_path": "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/cassava_disease_dataset_claude.json",
    "data_path": "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/course_project_dataset",
}

CASSAVA_DISEASE_DEEPSEEK_V3 = {
    "annotation_path": "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/cassava_disease_dataset_deepseek_V3.json",
    "data_path": "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/course_project_dataset",
}

CASSAVA_DISEASE_DEEPSEEK_R1 = {
    "annotation_path": "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/cassava_disease_dataset_deepseek_R1.json",
    "data_path": "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/course_project_dataset",
}

data_dict = {
    "cassava_disease_train": CASSAVA_DISEASE_TRAIN,
    "cassava_disease_test": CASSAVA_DISEASE_TEST,
    "cassava_disease_claude": CASSAVA_DISEASE_CLAUDE,
    "cassava_disease_deepseek_v3": CASSAVA_DISEASE_DEEPSEEK_V3,
    "cassava_disease_deepseek_r1": CASSAVA_DISEASE_DEEPSEEK_R1,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
