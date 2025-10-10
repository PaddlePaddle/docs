import argparse
import ast
import json
import os
import re
from collections import defaultdict


def get_pytorch_url(torch_api: str) -> str:
    """
    根据PyTorch API名称生成其官方文档URL

    Args:
        torch_api: PyTorch API的全限定名（如'torch.add', 'torch.nn.Linear', 'torch.Tensor.add'）

    Returns:
        对应API的官方文档URL字符串

    Rules:
    1. 优先检查特殊映射
    2. 优先检查是否有专门的generated页面
    3. 类方法指向父类页面#锚点
    4. 模块级函数/常量指向模块名.html
    5. Tensor相关API指向tensors.html
    6. 顶层函数（torch.xxx）指向torch.html
    7. 特殊处理torchvision等子库的URL结构
    """
    base_url = "https://pytorch.org/docs/stable/"
    torch_api = torch_api.replace(r"\_", "_")

    # 特殊映射：手动指定已知问题API的正确URL
    special_mappings = {
        "torch.cuda.check_error": "generated/torch.cuda.cudart.html",
        "torch.cuda.mem_get_info": "generated/torch.cuda.memory.mem_get_info.html",
        "torch.nn.attention.sdpa_kernel": "generated/torch.nn.attention.sdpa_kernel.html",
        "torch.torch.int32": "tensors.html#torch.int32",
        "torch.nn.attention._cur_sdpa_kernel_backends": "nn.attention.html#torch.nn.attention.sdpa_kernel",
        "torch.cuda.memory_reserved": "generated/torch.cuda.memory.memory_reserved.html",
        "torch.cuda.memory_allocated": "generated/torch.cuda.memory.memory_allocated.html",
        "torch.cuda.empty_cache": "generated/torch.cuda.memory.empty_cache.html",
    }

    # 检查特殊映射
    if torch_api in special_mappings:
        return f"{base_url}{special_mappings[torch_api]}"

    # 优先检查是否有专门的generated页面
    generated_apis = {
        "torch.pow": "generated/torch.pow.html",
        "torch.nn.utils.parameters_to_vector": "generated/torch.nn.utils.parameters_to_vector.html",
        "torch.nn.utils.vector_to_parameters": "generated/torch.nn.utils.vector_to_parameters.html",
        "torch.nn.Module": "generated/torch.nn.Module.html",
    }

    if torch_api in generated_apis:
        return f"{base_url}{generated_apis[torch_api]}"

    # 特殊处理：类方法（如torch.nn.Module.to）
    if torch_api.startswith("torch.nn.Module."):
        return f"{base_url}generated/torch.nn.Module.html#{torch_api}"

    if torch_api.startswith("torch.linalg.") or torch_api.startswith(
        "torch.cuda."
    ):
        return f"{base_url}generated/{torch_api}.html#{torch_api}"

    # 特殊子库处理（torchvision）
    if torch_api.startswith("torchvision."):
        vision_base = "https://pytorch.org/vision/stable/"
        if torch_api == "torchvision.models":
            return f"{vision_base}models.html"
        return f"{vision_base}generated/{torch_api}.html#{torch_api}"

    # 特殊处理：torch.__version__相关
    if torch_api.startswith("torch.__version__"):
        return base_url  # 版本信息通常在首页

    # 特殊处理：torch.distributed.ReduceOp枚举值
    if torch_api.startswith("torch.distributed.ReduceOp."):
        return f"{base_url}distributed.html#{torch_api}"

    # 特殊处理：torch.autograd.Function
    if torch_api == "torch.autograd.Function":
        return f"{base_url}autograd.html#{torch_api}"

    # 特殊处理：torch.utils.cpp_extension
    if torch_api.startswith("torch.utils.cpp_extension"):
        return f"{base_url}cpp_extension.html#{torch_api}"

    # 1. 处理Tensor相关API
    if torch_api.startswith("torch.Tensor") or torch_api == "torch.Tensor":
        return f"{base_url}tensors.html#{torch_api}"

    # 2. 处理顶层函数（无子模块）
    if len(torch_api.split(".")) == 2 and torch_api.startswith("torch."):
        # 检查是否有专门的generated页面
        generated_check = [
            "torch.pow",
            "torch.abs",
            "torch.add",
            "torch.sub",
            "torch.mul",
            "torch.div",
            "torch.exp",
            "torch.log",
            "torch.sin",
            "torch.cos",
            "torch.tan",
            "torch.sigmoid",
        ]

        if any(torch_api.startswith(prefix) for prefix in generated_check):
            return f"{base_url}generated/{torch_api}.html"
        return f"{base_url}torch.html#{torch_api}"

    # 分割API路径
    parts = torch_api.split(".")
    module_path = ".".join(parts[:-1])  # 模块路径
    item_name = parts[-1]  # 最后一项名称

    # 特殊处理：torch.functional函数
    if parts[0] == "torch" and parts[1] == "functional":
        return f"{base_url}torch.html#{torch_api}"

    # 3. 处理模块级函数/常量
    if parts[0] == "torch" and not parts[-1][0].isupper():
        # 特殊模块映射（基于官方文档结构）
        module_map = {
            "torch.nn.init": "nn.init.html",
            "torch.nn.functional": "nn.functional.html",
            "torch.cuda.amp": "amp.html",
            "torch.distributions": "distributions.html",
            "torch.nn.utils": "nn.utils.html",
            "torch.optim": "optim.html",
            "torch.random": "random.html",
            "torch.special": "special.html",
            "torch.distributed": "distributed.html",
            "torch.utils.data": "data.html",
        }
        module_key = ".".join(parts[:-1])
        module_slug = module_map.get(module_key, f"generated/{module_key}.html")

        # 检查是否是应该指向generated目录的API
        generated_modules = [
            "torch.nn.utils.parameters_to_vector",
            "torch.nn.utils.vector_to_parameters",
        ]

        if torch_api in generated_modules:
            return f"{base_url}generated/{torch_api}.html"

        return f"{base_url}{module_slug}#{torch_api}"

    # 4. 处理类/独立函数
    if parts[-1][0].isupper() or len(parts) == 1:
        # 特殊类映射
        class_map = {
            "torch.autograd.Function": "autograd.html",
            "torch.utils.cpp_extension.BuildExtension": "cpp_extension.html",
            "torch.nn.Module": "generated/torch.nn.Module.html",
        }
        if torch_api in class_map:
            return f"{base_url}{class_map[torch_api]}#{torch_api}"
        return f"{base_url}generated/{torch_api}.html#{torch_api}"

    # 5. 默认处理（类方法）
    # 特殊处理类方法
    class_method_map = {
        "torch.nn.Module": "generated/torch.nn.Module.html",
        "torch.utils.cpp_extension.BuildExtension": "cpp_extension.html",
    }

    for class_name, page_name in class_method_map.items():
        if module_path == class_name:
            return f"{base_url}{page_name}#{torch_api}"

    # 默认情况下，尝试生成到generated目录
    return f"{base_url}generated/{module_path}.html#{torch_api}"


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
    # mapping_url_head = "https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/"
    mapping_url_head = "https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/"
    # 定义两个可能的文档目录路径
    api_difference_dirs = [
        os.path.join(base_dir, "api_difference"),
    ]

    # 将torch_api中的特殊字符转换为下划线，并添加.md后缀，构成文件名
    expected_filename = f"{torch_api}.md"
    final_name = f"{torch_api}.html"

    for search_dir in api_difference_dirs:
        for root, dirs, files in os.walk(search_dir):
            if expected_filename in files:
                relative_path = os.path.relpath(
                    os.path.join(root, final_name), base_dir
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


def extract_no_need_convert_list(file_path):
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


def generate_category1_table(
    docs_mapping, no_need_convert_file_path, base_dir, existing_apis
):
    """
    生成类别1（API完全一致）的Markdown表格
    """
    white_list = []

    no_need_convert_list = extract_no_need_convert_list(
        no_need_convert_file_path
    )

    rows = []  # 存储表格行数据的列表
    used_apis = set()  # 用于记录已处理的API，避免重复

    # 处理no_need_convert_list中的每个Torch API
    for torch_api in no_need_convert_list:
        existing_apis.add(torch_api)
        if "__" in torch_api:
            torch_api = torch_api.replace("_", r"\_")
        if torch_api in used_apis:
            continue
        paddle_api = torch_api.replace("torch", "paddle")
        used_apis.add(torch_api)  # 标记该API已处理
        existing_apis.add(torch_api)

        src_url = get_pytorch_url(torch_api)
        dst_url = None
        # 构建第二列和第三列的字符串内容，包含URL（如果存在）
        col2 = f"[{torch_api}]({src_url})" if src_url else torch_api
        col3 = f"[{paddle_api}]({dst_url})" if dst_url else paddle_api
        rows.append((torch_api, col2, col3, "-"))

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
    attribute_mapping_file_path,
):
    """
    生成类别2（仅API调用方式不一致）的Markdown表格
    """
    whitelist_skip = [
        # "torch.Tensor.numel",# invoke_diff
        # "torch.Tensor.nelement",
        # "torch.Tensor.is_inference", # 不确定
        # "torch.numel", # 返回类型不一致
        # "torch.is_inference",
        # "torch.utils.data.WeightedRandomSampler",
        # "torch.utils.data.RandomSampler",
    ]

    no_need_convert_list = extract_no_need_convert_list(
        no_need_convert_file_path
    )

    # 加载api_mapping.json文件
    with open(api_mapping_file_path, "r", encoding="utf-8") as f:
        api_mapping_data = json.load(f)

    with open(attribute_mapping_file_path, "r", encoding="utf-8") as f:
        attribute_mapping_data = json.load(f)

    api_mapping_data = api_mapping_data | attribute_mapping_data

    rows = []  # 存储表格行数据的列表
    used_apis = set()  # 用于记录已处理的API，避免重复

    # 处理api_mapping中Matcher为"UnchangeMatcher"且不在no_need_convert_list中的API
    for src_api, mapping_info in api_mapping_data.items():
        if src_api in whitelist_skip or src_api in no_need_convert_list:
            continue
        matcher = mapping_info.get("Matcher", "")
        # ChangeAPIMatcher、TensorFunc2PaddleFunc、Func2Attribute、Attribute2Func类别
        if matcher in [
            "ChangeAPIMatcher",
            "TensorFunc2PaddleFunc",
            "Func2Attribute",
            "Attribute2Func",
            "NumelMatcher",
            "Is_InferenceMatcher",
        ]:
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
        if mapping_type in ["无参数", "参数完全一致", "仅 API 调用方式不一致"]:
            print(src_api)

            # 生成备注列的超链接
            # remark_link = get_mapping_doc_url(src_api, base_dir)
            # rows.append((src_api, col2, col3, remark_link))

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
        remark = f"``{torch_api_alias_display}`` 别名， {get_mapping_doc_url(torch_api_alias, base_dir)}"

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
    docs_mapping, md_content, base_dir, existing_apis
):
    """
    生成类别13（功能缺失）的Markdown表格
    直接从主文档中解析类别13的表格内容
    """
    # 定位类别13的表格
    pattern = r"### 13\. 功能缺失([\s\S]*?)(?=### |$)"
    match = re.search(pattern, md_content)
    if not match:
        print("未找到类别13的表格")

    section_content = match.group(1)

    # 解析表格内容
    table_pattern = r"\| 序号 \| Pytorch 最新 release \| Paddle develop \| 备注 \|\n\|[-\| ]+\|\n([\s\S]*?)(?=\n\n|\Z)"
    table_match = re.search(table_pattern, section_content)
    if not table_match:
        return ""

    table_content = table_match.group(1)
    rows = []

    # 解析每一行
    for line in table_content.split("\n"):
        if not line.startswith("|"):
            continue

        parts = line.split("|")
        if len(parts) < 5:
            continue

        # 提取各列内容
        torch_api_cell = parts[2].strip()
        paddle_api_cell = parts[3].strip()
        remark_cell = parts[4].strip()

        # 提取Torch API名称（处理超链接）
        torch_api_match = re.match(r"\[(.*?)\]\(.*?\)", torch_api_cell)
        torch_api = (
            torch_api_match.group(1) if torch_api_match else torch_api_cell
        )

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

        # 创建Torch API超链接（保留原链接）
        torch_link_match = re.search(r"\((.*?)\)", torch_api_cell)
        torch_url = torch_link_match.group(1) if torch_link_match else ""
        torch_display = (
            f"[{torch_api_display}]({torch_url})" if torch_url else torch_api
        )

        # 创建Paddle API超链接
        paddle_display = (
            f"[{dst_api_display}]({dst_api_url})" if dst_api_url else dst_api
        )

        # 保留原备注内容
        rows.append((torch_api, torch_display, paddle_display, remark_cell))
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
        # github_url = convert_to_github_url(api_md, base_dir)
        github_url = get_mapping_doc_url(api_name, base_dir)

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
        remark = f"{github_url}" if github_url else "-"

        # 添加表格行，并使用有效序号
        table_rows.append(
            f"| {valid_idx} | {torch_display} | {paddle_display} | {remark} |"
        )
        valid_idx += 1  # 序号递增

    # 构建完整的表格内容

    if len(table_rows) > 0:  # 如果存在有效行
        table_content = [
            "| 序号 | Pytorch 最新 release | Paddle develop | 备注 |",
            "|------|-------------------|---------------|------|",
            *table_rows,
        ]
    else:
        table_content = [
            "| 序号 | Pytorch 最新 release | Paddle develop | 备注 |",
            "|------|-------------------|---------------|------|",
            "新增中......",
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
        os.path.dirname(__file__), "api_difference_info.json"
    )
    no_need_convert_path = os.path.join(
        os.path.dirname(__file__), "global_var.py"
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
    attribute_mapping_path = os.path.join(
        os.path.dirname(__file__), "attribute_mapping.json"
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
        attribute_mapping_path,
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
        updated_content,
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
