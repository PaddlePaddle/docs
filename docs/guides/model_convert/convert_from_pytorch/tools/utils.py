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
        # return f"{base_url}torch.html#{torch_api}"
        return f"{base_url}generated/{torch_api}.html"

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
    return f"{base_url}generated/{torch_api}.html#{torch_api}"


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

    for category, apis in category_api_map.items():
        category_api_map[category] = sorted(
            apis, key=lambda x: x["api_name"].replace(r"\_", "_")
        )

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


def get_paddle_url(paddle_api: str) -> str:
    """
    根据Paddle API名称生成其官方文档URL

    Args:
        paddle_api: Paddle API的全限定名（如'paddle.distributed.reduce_scatter', 'paddle.linalg.det'）

    Returns:
        对应API的官方文档URL字符串
    """
    # 将API名称转换为URL格式
    # 例如: paddle.distributed.reduce_scatter -> paddle/distributed/reduce_scatter
    url_path = paddle_api.replace(".", "/") + "_cn.html"

    # 构建URL
    base_url = (
        "https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/"
    )
    url = base_url + url_path

    # 检查RST文件是否存在
    root_dir = "."  # 默认根目录
    rst_path = os.path.join(
        root_dir, "api", url_path.replace("_cn.html", ".rst")
    )

    if not os.path.exists(rst_path):
        print(
            f"Warning: RST file for {paddle_api} does not exist at {rst_path}"
        )

    # 添加锚点
    anchor = url_path.replace(
        ".html", "#cn-api-" + "-".join(paddle_api.split("."))
    )
    return url + "#" + anchor


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
