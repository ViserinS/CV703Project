import json
import os
import random

def split_cassava_dataset(json_file_path, output_dir, train_ratio=0.8, random_seed=42):
    """
    按照指定比例划分cassava数据集为训练集和测试集，只处理annotation文件
    
    参数:
    json_file_path (str): 数据集annotation文件的路径
    output_dir (str): 输出目录
    train_ratio (float): 训练集比例，默认0.8 (80%)
    random_seed (int): 随机种子，确保结果可复现
    """
    # 设置随机种子，确保结果可复现
    random.seed(random_seed)
    
    # 读取原始JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据集大小: {len(data)}个样本")
    
    # 随机打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 计算训练集大小
    train_size = int(len(shuffled_data) * train_ratio)
    
    # 划分数据集
    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]
    
    print(f"训练集大小: {len(train_data)}个样本 ({len(train_data)/len(data):.2%})")
    print(f"测试集大小: {len(test_data)}个样本 ({len(test_data)/len(data):.2%})")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集和测试集JSON文件
    train_json_path = os.path.join(output_dir, 'train_set.json')
    test_json_path = os.path.join(output_dir, 'test_set.json')
    
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"训练集JSON已保存到: {train_json_path}")
    print(f"测试集JSON已保存到: {test_json_path}")

    return train_data, test_data

if __name__ == "__main__":
    # 设置文件路径和参数
    json_file_path = "/remote-home/peachilk/codebase/Qwen2.5-VL/qwen-vl-finetune/dataset/cassava_disease_dataset_claude.json"  # 原始数据集JSON文件路径
    output_dir = "annotation"  # 输出目录
    train_ratio = 0.8  # 训练集比例
    
    # 执行划分
    train_data, test_data = split_cassava_dataset(
        json_file_path=json_file_path,
        output_dir=output_dir,
        train_ratio=train_ratio
    )
    
    # 输出一些统计信息 - 可以扩展以分析各类疾病在训练集和测试集中的分布
    print("\n数据集划分完成！")
    print(f"原始数据集: 486个样本")  # 根据您提供的信息
    print(f"训练集: {len(train_data)}个样本")
    print(f"测试集: {len(test_data)}个样本")