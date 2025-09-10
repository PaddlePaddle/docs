import os
import shutil
from pathlib import Path

def categorize_and_move_md_files(src_root, dest_root):
    """
    递归遍历源目录下的所有.md文件，根据第一行的差异类别将其移动到目标目录下对应的子目录中。

    Args:
        src_root (str): 源目录路径，包含.md文件（可能存在于子目录中）。
        dest_root (str): 目标根目录路径，所有按类别划分的子目录将创建于此。
    """
    # 定义差异类别中英文映射关系（中文 -> 英文目录名）
    category_mapping = {
        "仅参数名不一致": "parameter_name_only_different",
        "paddle 参数更多": "paddle_has_more_parameters",
        "参数默认值不一致": "parameter_default_value_different",
        "torch 参数更多": "torch_has_more_parameters",
        "输入参数用法不一致": "input_parameter_usage_different",
        "输入参数类型不一致": "input_parameter_type_different",
        "返回参数类型不一致": "return_parameter_type_different",
        "组合替代实现": "combination_alternative_implementation"
    }
    # 获取所有已知的英文类别目录名
    known_categories_en = set(category_mapping.values())

    # 确保目标根目录存在
    Path(dest_root).mkdir(parents=True, exist_ok=True)

    # 使用os.walk递归遍历源目录
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith('.md'):
                md_file_path = os.path.join(root, file)
                
                try:
                    # 读取.md文件的第一行
                    with open(md_file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                    
                    # 提取差异类别（假设类别信息在第一行，例如"## [ torch 参数更多 ]torch.signal.windows.blackman"）
                    # 这里尝试提取方括号[]内的内容，或者根据你的实际格式进行调整
                    extracted_category = None
                    if "[" in first_line and "]" in first_line:
                        # 尝试提取方括号内的内容
                        start_index = first_line.find("[") + 1
                        end_index = first_line.find("]", start_index)
                        if end_index != -1:
                            extracted_category = first_line[start_index:end_index].strip()
                    
                    # 如果没有通过方括号提取到，或者你的文件格式不同，可能需要其他解析方式
                    # 例如，如果第一行就是直接的类别描述，可以直接使用first_line
                    # 这里根据你的实际情况调整提取逻辑
                    # extracted_category = first_line  # 如果第一行直接是类别名

                    # 根据提取到的中文类别查找映射的英文目录名
                    target_category_dir = None
                    if extracted_category and extracted_category in category_mapping:
                        target_category_dir = category_mapping[extracted_category]
                    else:
                        # 如果提取的类别不在已知映射中，打印信息并跳过此文件
                        print(f"Warning: Unknown category '{extracted_category}' in file: {md_file_path}. Skipping.")
                        continue

                    # 构建目标目录路径
                    target_dir_path = os.path.join(dest_root, target_category_dir)
                    # 确保目标目录存在
                    Path(target_dir_path).mkdir(parents=True, exist_ok=True)
                    
                    # 构建目标文件路径
                    target_file_path = os.path.join(target_dir_path, file)
                    
                    # 移动文件（如果目标文件已存在，可能会覆盖，请谨慎操作）
                    shutil.move(md_file_path, target_file_path)
                    print(f"Moved: {md_file_path} -> {target_file_path}")
                
                except Exception as e:
                    print(f"Error processing file {md_file_path}: {e}")

# 使用示例
if __name__ == "__main__":
    source_directory = "../deprecated/api_difference_deprecated"  # 替换为你的源目录路径
    destination_directory = "../api_difference"                   # 替换为你的目标根目录路径
    
    categorize_and_move_md_files(source_directory, destination_directory)