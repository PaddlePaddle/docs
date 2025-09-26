import argparse
import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# =====================
# 常量定义
# =====================
# Markdown表格格式
TABLE_HEADER = "| 序号 | Pytorch 最新 release | Paddle develop | 备注 |"
TABLE_SEPARATOR = "|------|-------------------|---------------|------|"

# 类别名称列表
CATEGORY_NAMES = [
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

# 白名单列表
WHITELIST_SKIP = [
    "torch.Tensor.numel",
    "torch.Tensor.nelement",
    "torch.Tensor.is_inference",
    "torch.numel",
    "torch.is_inference",
    "torch.ge",
    "torch.utils.data.WeightedRandomSampler",
    "torch.utils.data.RandomSampler",
]

WHITELIST_NO_CONVERT = [
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

# 正则表达式模式
CATEGORY_TITLE_PATTERN = r"### \d*\.?\s*({})".format(
    "|".join(re.escape(c) for c in CATEGORY_NAMES)
)
API_PATTERN = r"##\s*\d*\.?\s*\[(.*?)\](.*)"
TABLE_PATTERN = r"\| 序号 \| Pytorch 最新 release \| Paddle develop \| 备注 \|\n\|[-\| ]+\|\n"

# 文件路径常量
BASE_DIR = Path(__file__).resolve().parent
API_DIFFERENCE_DIR = os.path.join(BASE_DIR, "api_difference")
API_MAPPING_JSON = os.path.join(BASE_DIR, "api_mapping.json")
API_ALIAS_MAPPING_JSON = os.path.join(BASE_DIR, "api_alias_mapping.json")
NO_NEED_CONVERT_PY = os.path.join(BASE_DIR, "global_var.py")
PYTORCH_API_MAPPING_MD = os.path.join(
    BASE_DIR.parent, "pytorch_api_mapping_cn.md"
)

# GitHub URL相关常量
GITHUB_BASE_URL = "https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/"
DOC_BASE_URL = "https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/"


# =====================
# 辅助函数
# =====================
def escape_underscores_in_api(api_name):
    r"""处理PyTorch API名称中的下划线转义。

    参数:
        api_name (str): 待处理的API名称字符串

    返回:
        str: 处理后的字符串。如果下划线出现次数>=2，则所有下划线被替换为'\_'；
             否则返回原字符串。
    """
    underscore_count = api_name.count("_")
    if underscore_count >= 2:
        return api_name.replace("_", r"\_")
    else:
        return api_name


def get_base_dir():
    """动态获取基础目录路径，确保代码可在任意位置执行"""
    return str(Path(__file__).resolve().parent)


def load_mapping_json(json_path):
    """加载docs_mapping.json文件"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"错误: 读取JSON文件 {json_path} 时出错: {e!s}")
        return []


def get_mapping_doc_url(torch_api, base_dir):
    """根据torch_api名称，递归查找对应的差异对比文档，并返回Markdown格式的超链接字符串"""
    # 定义文档目录路径
    api_difference_dirs = [API_DIFFERENCE_DIR]

    # 将torch_api中的特殊字符转换为下划线，并添加.md后缀，构成文件名
    expected_filename = f"{torch_api}.md"
    final_name = f"{torch_api}.html"

    for search_dir in api_difference_dirs:
        for root, dirs, files in os.walk(search_dir):
            if expected_filename in files:
                relative_path = os.path.relpath(
                    os.path.join(root, final_name), base_dir
                )
                full_url = DOC_BASE_URL + relative_path.replace(os.sep, "/")
                return f"[差异对比]({full_url})"

    return "-"


def parse_md_files(directories):
    """递归扫描目录中的所有.md文件，解析第一行获取类别和API名称

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
                        match = re.match(API_PATTERN, first_line)
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


def extract_no_need_convert_list(file_path):
    """从no_need_convert.txt文件中提取无需转换的API列表"""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    tree = ast.parse(content)
    no_need_list = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GlobalManager":
            for class_node in node.body:
                if isinstance(class_node, ast.Assign) and any(
                    target.id == "NO_NEED_CONVERT_LIST"
                    for target in class_node.targets
                ):
                    # 提取列表字面量
                    list_source = ast.get_source_segment(
                        content, class_node.value
                    )
                    no_need_list = ast.literal_eval(list_source)
                    break
    return no_need_list


def generate_table(rows):
    """生成Markdown表格

    参数:
        rows: 表格行数据列表，每行格式为 (torch_api, col2, col3, remark)

    返回:
        str: 生成的Markdown表格字符串
    """
    table_lines = [TABLE_HEADER, TABLE_SEPARATOR]

    for idx, (_, col2, col3, remark) in enumerate(rows, start=1):
        table_lines.append(f"| {idx} | {col2} | {col3} | {remark} |")

    return "\n".join(table_lines)


# =====================
# 生成特定类别的表格
# =====================
def generate_category1_table(
    docs_mapping, no_need_convert_file_path, base_dir, existing_apis
):
    """生成类别1（API完全一致）的Markdown表格"""
    no_need_convert_list = extract_no_need_convert_list(
        no_need_convert_file_path
    )
    rows = []
    used_apis = set()

    # 处理no_need_convert_list中的每个Torch API
    for torch_api in no_need_convert_list:
        if torch_api in used_apis:
            continue
        paddle_api = torch_api.replace("torch", "paddle")
        used_apis.add(torch_api)
        existing_apis.add(torch_api)

        # 在docs_mapping中查找当前torch_api对应的信息
        mapping_info = docs_mapping.get(torch_api, {})
        src_url = mapping_info.get("src_api_url")

        # 查找对应的paddle_api映射信息
        dst_url = None
        for item in docs_mapping.values():
            if item.get("dst_api") == paddle_api:
                dst_url = item.get("dst_api_url")
                break

        # 构建第二列和第三列
        torch_api = escape_underscores_in_api(torch_api)
        paddle_api = escape_underscores_in_api(paddle_api)
        col2 = f"[{torch_api}]({src_url})" if src_url else torch_api
        col3 = f"[{paddle_api}]({dst_url})" if dst_url else paddle_api
        rows.append((torch_api, col2, col3, "-"))

    # 遍历docs_mapping，查找满足条件的额外API对
    for src_api, item in docs_mapping.items():
        mapping_type = item.get("mapping_type", "")
        dst_api = item.get("dst_api", "")

        if (
            mapping_type in ["无参数", "参数完全一致"]
            and src_api.startswith("torch")
            and src_api not in used_apis
            and src_api not in WHITELIST_NO_CONVERT
        ):
            expected_paddle_api = src_api.replace("torch", "paddle")
            if "__" in src_api:
                src_api = src_api.replace("_", r"\_")
                dst_api = dst_api.replace("_", r"\_")

            if expected_paddle_api == dst_api:
                used_apis.add(src_api)
                existing_apis.add(src_api)

                src_url = item.get("src_api_url")
                dst_url = item.get("dst_api_url")

                src_api_display = escape_underscores_in_api(src_api)
                dst_api_display = escape_underscores_in_api(dst_api)

                col2 = f"[{src_api_display}]({src_url})" if src_url else src_api
                col3 = f"[{dst_api_display}]({dst_url})" if dst_url else dst_api
                rows.append((src_api, col2, col3, "-"))

    return generate_table(rows)


def generate_category2_table(
    docs_mapping,
    api_mapping_file_path,
    no_need_convert_file_path,
    base_dir,
    existing_apis,
):
    """生成类别2（仅API调用方式不一致）的Markdown表格"""
    with open(no_need_convert_file_path, "r", encoding="utf-8") as f:
        no_need_convert_list = [line.strip() for line in f if line.strip()]

    with open(api_mapping_file_path, "r", encoding="utf-8") as f:
        api_mapping_data = json.load(f)

    rows = []
    used_apis = set()

    # 处理api_mapping中Matcher为"UnchangeMatcher"且不在no_need_convert_list中的API
    for src_api, mapping_info in api_mapping_data.items():
        if src_api in WHITELIST_SKIP or src_api in no_need_convert_list:
            continue
        matcher = mapping_info.get("Matcher", "")
        if matcher == "UnchangeMatcher":
            docs_mapping_info = docs_mapping.get(src_api, {})
            src_url = docs_mapping_info.get("src_api_url")

            paddle_api = mapping_info.get("paddle_api")
            if not paddle_api:
                paddle_api = docs_mapping_info.get("dst_api", "")

            # 查找paddle_api对应的dst_api_url
            dst_url = None
            for item in docs_mapping.values():
                if item.get("dst_api") == paddle_api:
                    dst_url = item.get("dst_api_url")
                    break

            src_api_display = escape_underscores_in_api(src_api)
            paddle_api_display = escape_underscores_in_api(paddle_api)

            col2 = f"[{src_api_display}]({src_url})" if src_url else src_api
            col3 = (
                f"[{paddle_api_display}]({dst_url})" if dst_url else paddle_api
            )
            remark_link = get_mapping_doc_url(src_api, base_dir)
            rows.append((src_api, col2, col3, remark_link))
            used_apis.add(src_api)
            existing_apis.add(src_api)

    # 遍历docs_mapping，查找mapping_type为"无参数"或"参数完全一致"，且src_api替换后与dst_api不等的API
    for src_api, item in docs_mapping.items():
        mapping_type = item.get("mapping_type", "")
        dst_api = item.get("dst_api", "")

        if (
            src_api in WHITELIST_SKIP
            or src_api in no_need_convert_list
            or src_api in used_apis
        ):
            continue

        if mapping_type in ["无参数", "参数完全一致"] and "torch" in src_api:
            expected_paddle_api = src_api.replace("torch", "paddle")
            if expected_paddle_api != dst_api:
                used_apis.add(src_api)
                existing_apis.add(src_api)

                src_url = item.get("src_api_url")
                dst_url = item.get("dst_api_url")

                src_api_display = escape_underscores_in_api(src_api)
                dst_api_display = escape_underscores_in_api(dst_api)

                col2 = f"[{src_api_display}]({src_url})" if src_url else src_api
                col3 = f"[{dst_api_display}]({dst_url})" if dst_url else dst_api
                remark_link = get_mapping_doc_url(src_api, base_dir)
                rows.append((src_api, col2, col3, remark_link))

    return generate_table(rows)


def generate_api_alias_table(
    docs_mapping, api_alias_mapping_path, base_dir, existing_apis
):
    """生成类别12（API 别名映射）的Markdown表格"""
    try:
        with open(api_alias_mapping_path, "r", encoding="utf-8") as f:
            api_alias_data = json.load(f)
    except Exception as e:
        print(
            f"错误: 读取API别名映射文件 {api_alias_mapping_path} 时出错: {e!s}"
        )
        return ""

    rows = []
    used_apis = set()

    # 遍历api_alias_data，为每个别名映射生成表格行
    for torch_api, torch_api_alias in api_alias_data.items():
        if torch_api in existing_apis or torch_api_alias in existing_apis:
            continue

        mapping_info = docs_mapping.get(torch_api_alias, {})
        dst_api = mapping_info.get("dst_api", "-")
        dst_api_url = mapping_info.get("dst_api_url", "")

        src_api_url = docs_mapping.get(torch_api, {}).get("src_api_url", "")

        torch_api_display = escape_underscores_in_api(torch_api)
        torch_api_alias_display = torch_api_alias.replace(r"\_", "_")
        dst_api_display = escape_underscores_in_api(dst_api)

        torch_display = (
            f"[{torch_api_display}]({src_api_url})"
            if src_api_url
            else torch_api
        )
        paddle_display = (
            f"[{dst_api_display}]({dst_api_url})" if dst_api_url else dst_api
        )
        remark = f"``{torch_api_alias_display}`` 别名， {get_mapping_doc_url(torch_api_alias, base_dir)}"

        rows.append((torch_api, torch_display, paddle_display, remark))
        used_apis.add(torch_api)
        used_apis.add(torch_api_alias)
        existing_apis.add(torch_api)
        existing_apis.add(torch_api_alias)

    return generate_table(rows)


def generate_no_implement_table(
    md_content, docs_mapping, base_dir, existing_apis
):
    """生成类别13（功能缺失）的Markdown表格，从原始Markdown内容中提取第13类的表格"""
    # 从原始内容中提取第13类的表格
    pattern = r"### 13\. 功能缺失[\s\S]*?(\| 序号 \| Pytorch 最新 release \| Paddle develop \| 备注 \|\n\|[-\| ]+\|\n)[\s\S]*?"
    match = re.search(pattern, md_content)

    if not match:
        # 如果没有找到第13类的表格，返回空表格
        print("未找到第13类的表格")
        return generate_table([])

    # 提取表格内容
    table_content = match.group(1).strip()

    # 从表格中提取每一行
    rows = []
    for line in table_content.split("\n"):
        if line.strip() == "" or line.startswith("|------"):
            continue
        parts = [part.strip() for part in line.split("|")[1:-1]]
        if len(parts) >= 4:
            torch_api_with_url = parts[1]
            remark = parts[3]
            # 处理torch_api中的可能的链接
            torch_api = torch_api_with_url
            if (
                "[" in torch_api_with_url
                and "]" in torch_api_with_url
                and "(" in torch_api_with_url
                and ")" in torch_api_with_url
            ):
                torch_api = torch_api_with_url.split("[")[1].split("]")[0]
            rows.append((torch_api, torch_api_with_url, remark))

    # 生成新的表格
    new_rows = []
    for torch_api, torch_api_with_url, remark in rows:
        if torch_api in existing_apis:
            continue

        mapping_info = docs_mapping.get(torch_api, {})
        dst_api = mapping_info.get("dst_api", "-")
        dst_api_url = mapping_info.get("dst_api_url", "")

        torch_api_display = escape_underscores_in_api(torch_api)
        dst_api_display = escape_underscores_in_api(dst_api)

        torch_display = f"{torch_api_with_url}"
        paddle_display = (
            f"[{dst_api_display}]({dst_api_url})" if dst_api_url else dst_api
        )

        new_rows.append((torch_api, torch_display, paddle_display, remark))
        existing_apis.add(torch_api)

    return generate_table(new_rows)


# =====================
# 文档更新函数
# =====================
def update_mapping_table(
    md_content, category, api_list, mapping_data, existing_apis, base_dir
):
    """更新指定类别的映射表格，过滤掉未找到对应Paddle API的条目和重复API

    参数:
        md_content: 原始Markdown内容
        category: 类别名称
        api_list: API列表
        mapping_data: 映射数据
        existing_apis: 已处理的API集合
        base_dir: 基础目录路径

    返回:
        str: 更新后的Markdown内容
    """
    # 构建API名称到映射数据的字典
    api_mapping_dict = {}
    for item in mapping_data:
        src_api = item.get("src_api", "")
        api_mapping_dict[src_api] = {
            "dst_api": item.get("dst_api", ""),
            "src_api_url": item.get("src_api_url", ""),
            "dst_api_url": item.get("dst_api_url", ""),
        }

    # 生成表格行：仅处理能找到对应Paddle API且未重复的条目
    table_rows = []
    valid_idx = 1

    for api_info in api_list:
        api_name = api_info["api_name"]
        if api_name in existing_apis:
            continue

        mapping_info = api_mapping_dict.get(api_name, {})
        dst_api = mapping_info.get("dst_api", "-")

        if dst_api == "暂无" or not dst_api:
            dst_api = "-"

        src_api_url = mapping_info.get("src_api_url", "")
        dst_api_url = mapping_info.get("dst_api_url", "")
        github_url = get_mapping_doc_url(api_name, base_dir)

        api_name_display = escape_underscores_in_api(api_name)
        dst_api_display = escape_underscores_in_api(dst_api)

        torch_display = (
            f"[{api_name_display}]({src_api_url})" if src_api_url else api_name
        )
        paddle_display = (
            f"[{dst_api_display}]({dst_api_url})" if dst_api_url else dst_api
        )
        remark = f"{github_url}" if github_url else "-"

        table_rows.append((api_name, torch_display, paddle_display, remark))
        valid_idx += 1

    # 构建完整的表格内容
    if table_rows:
        table_content = generate_table(table_rows)
    else:
        table_content = "\n".join(
            [TABLE_HEADER, TABLE_SEPARATOR, "\n新增中......"]
        )

    # 替换原内容中的表格
    pattern = rf"({CATEGORY_TITLE_PATTERN}).*?{TABLE_PATTERN}[\s\S]*?(?=### \d*\.?\s*|\Z)"
    replacement = rf"\1{table_content}\n\n"
    return re.sub(pattern, replacement, md_content, flags=re.MULTILINE)


def add_category_numbers(md_content, all_categories):
    """为所有类别标题添加序号（1~13）"""
    updated_content = md_content
    for idx, category in enumerate(all_categories, 1):
        pattern = rf"### \d*\.?\s*{re.escape(category)}"
        replacement = f"### {idx}. {category}"
        updated_content = re.sub(pattern, replacement, updated_content)
    return updated_content


def update_special_category_table(md_content, category, table_content):
    """更新特殊类别（1和2）的表格内容"""
    pattern = rf"({CATEGORY_TITLE_PATTERN}).*?{TABLE_PATTERN}[\s\S]*?(?=### \d*\.?\s*|\Z)"
    replacement = rf"\1{table_content}\n\n"
    return re.sub(pattern, replacement, md_content, flags=re.MULTILINE)


# =====================
# 主函数
# =====================
def main():
    parser = argparse.ArgumentParser(
        description="更新PyTorch到PaddlePaddle API映射文档"
    )
    parser.add_argument(
        "--check", action="store_true", help="检查模式，输出到临时文件"
    )
    args = parser.parse_args()

    base_dir = get_base_dir()

    # 读取原始MD文件
    try:
        with open(PYTORCH_API_MAPPING_MD, "r", encoding="utf-8") as f:
            original_content = f.read()
    except Exception as e:
        print(f"错误: 读取Markdown文件 {PYTORCH_API_MAPPING_MD} 时出错: {e!s}")
        return

    # 加载映射JSON数据
    mapping_data = load_mapping_json(
        os.path.join(base_dir, "api_difference_info.json")
    )
    docs_mapping = (
        {item["src_api"]: item for item in mapping_data} if mapping_data else {}
    )

    # 为所有类别标题添加序号（1~13）
    updated_content = add_category_numbers(original_content, CATEGORY_NAMES)

    # 生成类别1和类别2的表格
    existing_apis = set()
    category1_table = generate_category1_table(
        docs_mapping, NO_NEED_CONVERT_PY, base_dir, existing_apis
    )
    category2_table = generate_category2_table(
        docs_mapping,
        API_MAPPING_JSON,
        NO_NEED_CONVERT_PY,
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
    category_api_map = parse_md_files([API_DIFFERENCE_DIR])

    # 更新内容（只处理3-11类）
    for idx, category in enumerate(CATEGORY_NAMES, 1):
        if idx >= 3 and idx <= 11 and category in category_api_map:
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
        API_ALIAS_MAPPING_JSON,
        base_dir,
        existing_apis,
    )
    updated_content = update_special_category_table(
        updated_content, "API 别名", category12_table
    )

    # 生成类别13（功能缺失）的表格
    category13_table = generate_no_implement_table(
        updated_content,
        docs_mapping,
        base_dir,
        existing_apis,
    )
    updated_content = update_special_category_table(
        updated_content, "功能缺失", category13_table
    )

    # 确定输出文件
    output_file = (
        os.path.join(os.path.dirname(__file__), "tmp_check.md")
        if args.check
        else PYTORCH_API_MAPPING_MD
    )

    # 写入更新后的内容
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"成功: 文档已更新到 {output_file}")
    except Exception as e:
        print(f"错误: 写入文件 {output_file} 时出错: {e!s}")


if __name__ == "__main__":
    main()
