import argparse
import json
import os
import re
from collections import defaultdict


def escape_underscores_in_api(api_name):
    r"""
    处理PyTorch API名称中的下划线转义。

    参数:
        api_name (str): 待处理的API名称字符串

    返回:
        str: 处理后的字符串。如果下划线出现次数>=2，则所有下划线被替换为'\_'；
             否则返回原字符串。
    """
    # 统计下划线在字符串中出现的次数
    underscore_count = api_name.count("_")

    # 如果下划线出现次数大于等于2，则进行替换
    if underscore_count >= 2:
        return api_name.replace("_", r"\_")
    else:
        return api_name


def get_base_dir():
    """
    动态获取基础目录路径，确保代码可在任意位置执行
    """
    current_script_path = os.path.abspath(__file__)
    tools_dir = os.path.dirname(current_script_path)
    base_dir = os.path.dirname(tools_dir)  # 上一级目录
    return base_dir


def parse_md_files(directories):
    """
    递归扫描目录中的所有.md文件，解析第一行获取类别和API名称
    忽略标题中的序号（如"1. "），只提取纯类别名称
    """
    category_api_map = defaultdict(list)

    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".md"):
                    md_path = os.path.join(root, file)
                    try:
                        with open(md_path, "r", encoding="utf-8") as f:
                            first_line = f.readline().strip()

                        # 解析第一行格式：## [类别]API名称，忽略可能存在的序号
                        match = re.match(
                            r"##\s*\d*\.?\s*\[(.*?)\](.*)", first_line
                        )
                        if match:
                            category = match.group(1).strip()
                            api_name = match.group(2).strip()
                            # 只处理3-12类，前两类从主文档表格中提取
                            if category not in [
                                "API 完全一致",
                                "仅 API 调用方式不一致",
                            ]:
                                category_api_map[category].append(
                                    {
                                        "api_name": api_name.replace(
                                            r"\_", "_"
                                        ),
                                        "file_path": md_path,
                                    }
                                )
                        else:
                            print(
                                f"警告: 无法解析文件 {md_path} 的第一行: {first_line}"
                            )
                    except Exception as e:
                        print(f"错误: 读取文件 {md_path} 时出错: {e!s}")

    return category_api_map


def load_mapping_json(json_path):
    """
    加载docs_mapping.json文件
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"错误: 读取JSON文件 {json_path} 时出错: {e!s}")
        return []


def convert_to_github_url(local_path, base_dir):
    """
    将本地文件路径转换为GitHub URL
    """
    # 查找convert_from_pytorch在路径中的位置
    pattern = r".*(docs/guides/model_convert/convert_from_pytorch/.*)"
    match = re.search(pattern, local_path)
    if match:
        relative_path = match.group(1)
        return (
            f"https://github.com/PaddlePaddle/docs/tree/develop/{relative_path}"
        )

    # 如果正则匹配失败，尝试基于基础目录构建相对路径
    try:
        relative_path = os.path.relpath(local_path, base_dir)
        return f"https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/{relative_path}"
    except:
        return ""


def get_mapping_doc_url(torch_api, base_dir):
    """
    根据torch_api名称，递归查找对应的差异对比文档，并返回Markdown格式的超链接字符串。
    """
    mapping_url_head = "https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/"

    # 定义两个可能的文档目录路径
    api_difference_dirs = [
        os.path.join(base_dir, "api_difference"),
        os.path.join(base_dir, "api_difference_third_party"),
    ]

    # 将torch_api中的特殊字符转换为下划线，并添加.md后缀，构成文件名
    expected_filename = f"{torch_api}.md"

    for search_dir in api_difference_dirs:
        for root, dirs, files in os.walk(search_dir):
            if expected_filename in files:
                relative_path = os.path.relpath(
                    os.path.join(root, expected_filename), base_dir
                )
                full_url = mapping_url_head + relative_path.replace(os.sep, "/")
                return f"[差异对比]({full_url})"

    return "-"


def parse_special_category_apis(md_content, category):
    """
    从主MD文档中解析特殊类别（API完全一致、仅API调用方式不一致）的API
    返回该类别中所有Torch API的集合
    """
    apis = set()
    lines = md_content.split("\n")
    in_target_section = False
    in_table = False

    for line in lines:
        # 检查是否进入目标类别部分
        if re.match(rf"### \d*\.?\s*{re.escape(category)}", line):
            in_target_section = True
            continue

        # 如果已在目标部分，检查是否进入表格
        if in_target_section:
            if re.match(r"\| 序号 \|", line):
                in_table = True
                continue

            # 检查是否离开目标部分
            if re.match(r"### \d*\.?\s*", line) and not re.match(
                rf"### \d*\.?\s*{re.escape(category)}", line
            ):
                break

            # 处理表格行
            if in_table and re.match(r"\| \d+ \|", line):
                parts = line.split("|")
                if len(parts) >= 3:
                    api_cell = parts[2].strip()  # 第三列是Torch API
                    # 提取API名称（处理超链接）
                    api_match = re.match(r"\[(.*?)\]\(.*?\)", api_cell)
                    api_name = api_match.group(1) if api_match else api_cell
                    if api_name:
                        apis.add(api_name)

    return apis


def generate_category1_table(
    docs_mapping, no_need_convert_file_path, base_dir, existing_apis
):
    """
    生成类别1（API完全一致）的Markdown表格
    """
    white_list = [
        "torch.Tensor.imag",
        "torch.Tensor.is_coalesced",
        "torch.Tensor.is_sparse",
        "torch.Tensor.is_sparse_csr",
        "torch.Tensor.logical_not_",
        "torch.Tensor.real",
        "torch.iinfo",
        "torch.nn.utils.clip_grad_norm_",
        "torch.nn.utils.clip_grad_value_",
    ]

    # 读取no_need_convert.txt文件，获取无需转换的API列表
    with open(no_need_convert_file_path, "r", encoding="utf-8") as f:
        no_need_convert_list = [line.strip() for line in f if line.strip()]

    rows = []  # 存储表格行数据的列表
    used_apis = set()  # 用于记录已处理的API，避免重复

    # 处理no_need_convert_list中的每个Torch API
    for torch_api in no_need_convert_list:
        if "__" in torch_api:
            torch_api = torch_api.replace("_", r"\_")
        if torch_api in used_apis:
            continue
        paddle_api = torch_api.replace("torch", "paddle")
        used_apis.add(torch_api)  # 标记该API已处理
        existing_apis.add(torch_api)

        # 在docs_mapping中查找当前torch_api对应的信息
        mapping_info = docs_mapping.get(torch_api, {})
        src_url = mapping_info.get("src_api_url") if mapping_info else None

        # 查找对应的paddle_api映射信息（可能需要遍历所有值）
        dst_url = None
        for item in docs_mapping.values():
            if item.get("dst_api") == paddle_api:
                dst_url = item.get("dst_api_url")
                break

        # 构建第二列和第三列的字符串内容，包含URL（如果存在）
        col2 = f"[{torch_api}]({src_url})" if src_url else torch_api
        col3 = f"[{paddle_api}]({dst_url})" if dst_url else paddle_api
        rows.append((torch_api, col2, col3, "-"))

    # 遍历docs_mapping，查找满足条件的额外API对
    for src_api, item in docs_mapping.items():
        mapping_type = item.get("mapping_type", "")
        dst_api = item.get("dst_api", "")

        # 检查条件：mapping_type为"无参数"或"参数完全一致"，且src_api以"torch"开头，替换后与dst_api相同，且不在no_need_convert_list中
        if (mapping_type in ["无参数", "参数完全一致"]) and src_api.startswith(
            "torch"
        ):
            expected_paddle_api = src_api.replace("torch", "paddle")
            if "__" in src_api:
                src_api = src_api.replace("_", r"\_")
                dst_api = dst_api.replace("_", r"\_")
            if (
                expected_paddle_api == dst_api
                and src_api not in used_apis
                and src_api not in white_list
            ):
                used_apis.add(src_api)  # 标记该API已处理
                existing_apis.add(src_api)

                src_url = item.get("src_api_url")
                dst_url = item.get("dst_api_url")

                src_api_display = escape_underscores_in_api(src_api)
                dst_api_display = escape_underscores_in_api(dst_api)

                col2 = f"[{src_api_display}]({src_url})" if src_url else src_api
                col3 = f"[{dst_api_display}]({dst_url})" if dst_url else dst_api
                rows.append((src_api, col2, col3, "-"))

    # 生成Markdown表格字符串
    table_lines = [
        "| 序号 | Pytorch 最新 release | Paddle develop | 备注 |",
        "|------|-------------------|---------------|------|",
    ]

    for idx, (_, col2, col3, remark) in enumerate(rows, start=1):
        table_lines.append(f"| {idx} | {col2} | {col3} | {remark} |")

    return "\n".join(table_lines)


def generate_category2_table(
    docs_mapping,
    api_mapping_file_path,
    no_need_convert_file_path,
    base_dir,
    existing_apis,
):
    """
    生成类别2（仅API调用方式不一致）的Markdown表格
    """
    whitelist_skip = [
        "torch.Tensor.numel",
        "torch.Tensor.nelement",
        "torch.Tensor.is_inference",
        "torch.numel",
        "torch.is_inference",
        "torch.ge",
        "torch.utils.data.WeightedRandomSampler",
        "torch.utils.data.RandomSampler",
    ]

    # 读取no_need_convert.txt文件，获取无需转换的API列表
    with open(no_need_convert_file_path, "r", encoding="utf-8") as f:
        no_need_convert_list = [line.strip() for line in f if line.strip()]

    # 加载api_mapping.json文件
    with open(api_mapping_file_path, "r", encoding="utf-8") as f:
        api_mapping_data = json.load(f)

    rows = []  # 存储表格行数据的列表
    used_apis = set()  # 用于记录已处理的API，避免重复

    # 处理api_mapping中Matcher为"UnchangeMatcher"且不在no_need_convert_list中的API
    for src_api, mapping_info in api_mapping_data.items():
        if src_api in whitelist_skip or src_api in no_need_convert_list:
            continue
        matcher = mapping_info.get("Matcher", "")
        if matcher == "UnchangeMatcher":
            # 在docs_mapping中查找当前src_api对应的信息
            docs_mapping_info = docs_mapping.get(src_api, {})
            src_url = docs_mapping_info.get("src_api_url")

            # 获取paddle_api，可能来自api_mapping或docs_mapping
            paddle_api = mapping_info.get("paddle_api")
            if not paddle_api:
                paddle_api = docs_mapping_info.get("dst_api", "")

            # 查找paddle_api对应的dst_api_url（可能需要遍历docs_mapping的值）
            dst_url = None
            for item in docs_mapping.values():
                if item.get("dst_api") == paddle_api:
                    dst_url = item.get("dst_api_url")
                    break

            src_api_display = escape_underscores_in_api(src_api)
            paddle_api_display = escape_underscores_in_api(paddle_api)
            # 构建第二列和第三列的字符串内容，包含URL（如果存在）
            col2 = f"[{src_api_display}]({src_url})" if src_url else src_api
            col3 = (
                f"[{paddle_api_display}]({dst_url})" if dst_url else paddle_api
            )

            # 生成备注列的超链接
            remark_link = get_mapping_doc_url(src_api, base_dir)
            rows.append((src_api, col2, col3, remark_link))
            used_apis.add(src_api)  # 标记该API已处理
            existing_apis.add(src_api)

    # 遍历docs_mapping，查找mapping_type为"无参数"或"参数完全一致"，且src_api替换后与dst_api不等的API
    for src_api, item in docs_mapping.items():
        mapping_type = item.get("mapping_type", "")
        dst_api = item.get("dst_api", "")
        if (
            src_api in whitelist_skip
            or src_api in no_need_convert_list
            or src_api in used_apis
        ):
            continue
        # 检查条件：mapping_type为"无参数"或"参数完全一致"，src_api包含"torch"，替换后与dst_api不等
        if (mapping_type in ["无参数", "参数完全一致"]) and "torch" in src_api:
            expected_paddle_api = src_api.replace("torch", "paddle")
            if expected_paddle_api != dst_api:
                used_apis.add(src_api)  # 标记该API已处理
                existing_apis.add(src_api)

                src_url = item.get("src_api_url")
                dst_url = item.get("dst_api_url")

                src_api_display = escape_underscores_in_api(src_api)
                dst_api_display = escape_underscores_in_api(dst_api)

                col2 = f"[{src_api_display}]({src_url})" if src_url else src_api
                col3 = f"[{dst_api_display}]({dst_url})" if dst_url else dst_api

                # 生成备注列的超链接
                remark_link = get_mapping_doc_url(src_api, base_dir)
                rows.append((src_api, col2, col3, remark_link))

    # 生成Markdown表格字符串
    table_lines = [
        "| 序号 | Pytorch 最新 release | Paddle develop | 备注 |",
        "|------|-------------------|---------------|------|",
    ]

    for idx, (_, col2, col3, remark) in enumerate(rows, start=1):
        table_lines.append(f"| {idx} | {col2} | {col3} | {remark} |")

    return "\n".join(table_lines)


def generate_api_alias_table(
    docs_mapping, api_alias_mapping_path, base_dir, existing_apis
):
    """
    生成类别12（API 别名映射）的Markdown表格
    """
    # 读取api_alias_mapping.json文件
    try:
        with open(api_alias_mapping_path, "r", encoding="utf-8") as f:
            api_alias_data = json.load(f)
    except Exception as e:
        print(
            f"错误: 读取API别名映射文件 {api_alias_mapping_path} 时出错: {e!s}"
        )
        return ""

    rows = []  # 存储表格行数据的列表
    used_apis = set()  # 用于记录已处理的API，避免重复

    # 遍历api_alias_data，为每个别名映射生成表格行
    for torch_api, torch_api_alias in api_alias_data.items():
        # 检查API是否已经在前面的类别中处理过
        if torch_api in existing_apis or torch_api_alias in existing_apis:
            continue

        # 在docs_mapping中查找torch_api_alias对应的Paddle API
        mapping_info = docs_mapping.get(torch_api_alias, {})
        dst_api = mapping_info.get("dst_api", "-")
        dst_api_url = mapping_info.get("dst_api_url", "")

        # 获取torch_api的URL
        src_api_url = docs_mapping.get(torch_api, {}).get("src_api_url", "")

        # 构建显示的API名称
        torch_api_display = escape_underscores_in_api(torch_api)
        torch_api_alias_display = torch_api_alias.replace(r"\_", "_")
        dst_api_display = escape_underscores_in_api(dst_api)

        # 创建Torch API超链接
        torch_display = (
            f"[{torch_api_display}]({src_api_url})"
            if src_api_url
            else torch_api
        )

        # 创建Paddle API超链接
        paddle_display = (
            f"[{dst_api_display}]({dst_api_url})" if dst_api_url else dst_api
        )

        # 构建备注列，格式为"{torch_api_alias}别名+[差异对比]{url}"
        remark = f"``{torch_api_alias_display}`` 别名, [{get_mapping_doc_url(torch_api_alias, base_dir)}]"

        # 添加表格行
        rows.append((torch_api, torch_display, paddle_display, remark))
        used_apis.add(torch_api)
        used_apis.add(torch_api_alias)
        existing_apis.add(torch_api)
        existing_apis.add(torch_api_alias)

    # 生成Markdown表格字符串
    table_lines = [
        "| 序号 | Pytorch 最新 release | Paddle develop | 备注 |",
        "|------|-------------------|---------------|------|",
    ]

    for idx, (_, col2, col3, remark) in enumerate(rows, start=1):
        table_lines.append(f"| {idx} | {col2} | {col3} | {remark} |")

    return "\n".join(table_lines)


def generate_no_implement_table(
    docs_mapping, no_implement_path, base_dir, existing_apis
):
    """
    生成类别13（功能缺失）的Markdown表格
    """
    # 读取no_implement.md文件
    try:
        with open(no_implement_path, "r", encoding="utf-8") as f:
            no_implement_content = f.read()
    except Exception as e:
        print(f"错误: 读取功能缺失文件 {no_implement_path} 时出错: {e!s}")
        return ""

    # 从no_implement_content中提取API信息
    # 使用正则表达式匹配NOT-IMPLEMENTED-ITEM
    pattern = r"NOT-IMPLEMENTED-ITEM\(`([^`]+)`, (https://pytorch\.org/docs/[^,]+), (.*)\)"
    matches = re.findall(pattern, no_implement_content, re.MULTILINE)
    # print(matches)
    rows = []  # 存储表格行数据的列表

    for idx, match in enumerate(matches, start=1):
        torch_api = match[0]
        torch_api_url = match[1]
        remark = match[2]
        # print(torch_api)
        # 检查API是否已经在前面的类别中处理过
        if torch_api in existing_apis:
            continue

        # 在docs_mapping中查找对应的Paddle API
        mapping_info = docs_mapping.get(torch_api, {})
        dst_api = mapping_info.get("dst_api", "-")
        dst_api_url = mapping_info.get("dst_api_url", "")

        # 构建显示的API名称
        torch_api_display = escape_underscores_in_api(torch_api)
        dst_api_display = escape_underscores_in_api(dst_api)

        # 创建Torch API超链接
        torch_display = f"[{torch_api_display}]({torch_api_url})"

        # 创建Paddle API超链接
        paddle_display = (
            f"[{dst_api_display}]({dst_api_url})" if dst_api_url else dst_api
        )

        # 构建备注列
        # 备注列已经包含在remark中了
        # if not remark.startswith("["):
        #     remark = f"[{remark}]"

        rows.append((torch_api, torch_display, paddle_display, remark))
        existing_apis.add(torch_api)

    # 生成Markdown表格字符串
    table_lines = [
        "| 序号 | Pytorch 最新 release | Paddle develop | 备注 |",
        "|------|-------------------|---------------|------|",
    ]

    for idx, (_, col2, col3, remark) in enumerate(rows, start=1):
        table_lines.append(f"| {idx} | {col2} | {col3} | {remark} |")

    return "\n".join(table_lines)


def update_mapping_table(
    md_content, category, api_list, mapping_data, existing_apis, base_dir
):
    """
    更新指定类别的映射表格，过滤掉未找到对应Paddle API的条目和重复API。
    """
    # 构建API名称到映射数据的字典
    api_mapping_dict = {}
    for item in mapping_data:
        src_api = item.get("src_api", "")
        api_mapping_dict[src_api] = {
            "dst_api": item.get("dst_api", ""),
            "src_api_url": item.get("src_api_url", ""),
            "dst_api_url": item.get("dst_api_url", ""),
            "source_file": item.get("source_file", ""),
        }

    # 生成表格行：仅处理能找到对应Paddle API且未重复的条目
    table_rows = []
    valid_idx = 1  # 有效序号计数器
    for api_info in api_list:
        api_name = api_info["api_name"]
        api_md = api_info["file_path"]

        # 检查API是否已在前两个特殊类别中存在
        if api_name in existing_apis:
            continue

        mapping_info = api_mapping_dict.get(api_name, {})
        dst_api = mapping_info.get("dst_api", "-")

        if dst_api == "暂无" or not dst_api:
            dst_api = "-"

        # 获取URL信息
        src_api_url = mapping_info.get("src_api_url", "")
        dst_api_url = mapping_info.get("dst_api_url", "")
        github_url = convert_to_github_url(api_md, base_dir)

        api_name_display = escape_underscores_in_api(api_name)
        dst_api_display = escape_underscores_in_api(dst_api)

        # 创建Torch API超链接
        torch_display = (
            f"[{api_name_display}]({src_api_url})" if src_api_url else api_name
        )

        # 创建Paddle API超链接
        paddle_display = (
            f"[{dst_api_display}]({dst_api_url})" if dst_api_url else dst_api
        )

        # 创建备注列内容
        remark = f"[差异对比]({github_url})" if github_url else "-"

        # 添加表格行，并使用有效序号
        table_rows.append(
            f"| {valid_idx} | {torch_display} | {paddle_display} | {remark} |"
        )
        valid_idx += 1  # 序号递增

    # 构建完整的表格内容
    if table_rows:  # 如果存在有效行
        table_content = [
            "| 序号 | Pytorch 最新 release | Paddle develop | 备注 |",
            "|------|-------------------|---------------|------|",
            *table_rows,
        ]
    else:
        table_content = [
            "| 序号 | Pytorch 最新 release | Paddle develop | 备注 |",
            "|------|-------------------|---------------|------|",
            "\n新增中......",
        ]

    table_content_str = "\n".join(table_content)

    # 替换原内容中的表格（考虑可能有序号的标题）
    # 添加额外的换行符确保格式正确
    pattern = rf"(### \d*\.?\s*{re.escape(category)}[\s\S]*?)(\| 序号 \| Pytorch 最新 release \| Paddle develop \| 备注 \|\n\|[-\| ]+\|\n)[\s\S]*?(?=### \d*\.?\s*|\Z)"
    replacement = rf"\1{table_content_str}\n\n"
    return re.sub(pattern, replacement, md_content, flags=re.MULTILINE)


def add_category_numbers(md_content, all_categories):
    """
    为所有类别标题添加序号（1~13）
    如果标题已有序号，会先移除旧序号再添加新序号
    """
    updated_content = md_content
    for idx, category in enumerate(all_categories, 1):
        # 先移除可能存在的旧序号
        pattern = rf"### \d*\.?\s*{re.escape(category)}"
        replacement = f"### {idx}. {category}"
        updated_content = re.sub(pattern, replacement, updated_content)
    return updated_content


def update_special_category_table(md_content, category, table_content):
    """
    更新特殊类别（1和2）的表格内容
    """
    # 更精确的正则表达式，确保只匹配特定类别的表格
    pattern = rf"(### \d*\.?\s*{re.escape(category)}[\s\S]*?)(\| 序号 \| Pytorch 最新 release \| Paddle develop \| 备注 \|\n\|[-\| ]+\|\n)[\s\S]*?(?=### \d*\.?\s*|\Z)"
    # 替换为：标题 + 新表格内容
    replacement = rf"\1{table_content}\n\n"
    return re.sub(pattern, replacement, md_content, flags=re.MULTILINE)


def main():
    parser = argparse.ArgumentParser(
        description="更新PyTorch到PaddlePaddle API映射文档"
    )
    parser.add_argument(
        "--check", action="store_true", help="检查模式，输出到临时文件"
    )
    args = parser.parse_args()

    # 获取基础目录
    base_dir = get_base_dir()

    # 定义路径
    md_file_path = os.path.join(base_dir, "pytorch_api_mapping_cn.md")
    json_file_path = os.path.join(
        os.path.dirname(__file__), "docs_mappings.json"
    )
    no_need_convert_path = os.path.join(
        os.path.dirname(__file__), "no_need_convert.txt"
    )
    api_mapping_path = os.path.join(
        os.path.dirname(__file__), "api_mapping.json"
    )
    api_alias_mapping_path = os.path.join(
        os.path.dirname(__file__), "api_alias_mapping.json"
    )
    no_implement_path = os.path.join(
        os.path.dirname(__file__), "no_implement.md"
    )

    api_dirs = [
        os.path.join(base_dir, "api_difference"),
        os.path.join(base_dir, "api_difference_third_party"),
    ]

    # 读取原始MD文件
    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            original_content = f.read()
    except Exception as e:
        print(f"错误: 读取Markdown文件 {md_file_path} 时出错: {e!s}")
        return

    # 加载映射JSON数据
    mapping_data = load_mapping_json(json_file_path)
    docs_mapping = (
        {item["src_api"]: item for item in mapping_data} if mapping_data else {}
    )

    # 定义所有可能的类别（13个类别）
    all_categories = [
        "API 完全一致",  # 序号1
        "仅 API 调用方式不一致",  # 序号2
        "仅参数名不一致",  # 序号3
        "paddle 参数更多",  # 序号4
        "参数默认值不一致",  # 序号5
        "torch 参数更多",  # 序号6
        "输入参数用法不一致",  # 序号7
        "输入参数类型不一致",  # 序号8
        "返回参数类型不一致",  # 序号9
        "组合替代实现",  # 序号10
        "可删除",  # 序号11
        "API 别名",  # 序号12
        "功能缺失",  # 序号13
    ]

    # 为所有类别标题添加序号（1~13）
    updated_content = add_category_numbers(original_content, all_categories)

    # 生成类别1和类别2的表格
    existing_apis = set()
    category1_table = generate_category1_table(
        docs_mapping, no_need_convert_path, base_dir, existing_apis
    )
    category2_table = generate_category2_table(
        docs_mapping,
        api_mapping_path,
        no_need_convert_path,
        base_dir,
        existing_apis,
    )

    updated_content = update_special_category_table(
        updated_content, "API 完全一致", category1_table
    )
    updated_content = update_special_category_table(
        updated_content, "仅 API 调用方式不一致", category2_table
    )

    print(
        f"信息: 从前两个特殊类别中总共解析出 {len(existing_apis)} 个API用于重复检查"
    )

    # 解析MD文件获取类别和API信息（3-11类）
    category_api_map = parse_md_files(api_dirs)

    # 更新内容（只处理3-11类）
    for idx, category in enumerate(all_categories, 1):
        if (
            idx >= 3 and idx <= 11 and category in category_api_map
        ):  # 只处理3-11类
            updated_content = update_mapping_table(
                updated_content,
                category,
                category_api_map[category],
                mapping_data,
                existing_apis,
                base_dir,
            )

    # 生成类别12（API 别名映射）的表格
    category12_table = generate_api_alias_table(
        docs_mapping,
        api_alias_mapping_path,
        base_dir,
        existing_apis,
    )
    updated_content = update_special_category_table(
        updated_content, "API 别名", category12_table
    )

    # 生成类别13（功能缺失）的表格
    category13_table = generate_no_implement_table(
        docs_mapping,
        no_implement_path,
        base_dir,
        existing_apis,
    )
    updated_content = update_special_category_table(
        updated_content, "功能缺失", category13_table
    )

    # 确定输出文件
    if args.check:
        output_file = os.path.join(os.path.dirname(__file__), "tmp_check.md")
    else:
        output_file = md_file_path

    # 写入更新后的内容
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"成功: 文档已更新到 {output_file}")
    except Exception as e:
        print(f"错误: 写入文件 {output_file} 时出错: {e!s}")


if __name__ == "__main__":
    main()
