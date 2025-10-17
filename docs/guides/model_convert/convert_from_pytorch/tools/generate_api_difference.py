import argparse
import ast
import inspect
import os
import subprocess

from utils import get_paddle_url, get_pytorch_url, load_mapping_json


def get_function_signature(api_name: str, framework: str) -> str:
    """
    获取API的函数签名

    Args:
        api_name: API的全限定名
        framework: 'torch' 或 'paddle'

    Returns:
        函数签名字符串
    """
    # 导入框架
    if framework == "torch":
        import torch

        module = torch
    elif framework == "paddle":
        import paddle

        module = paddle
    elif framework == "fairscale":
        import fairscale

        module = fairscale
    elif framework == "transformers":
        import transformers

        module = transformers
    elif framework == "torchvision":
        import torchvision

        module = torchvision
    elif framework == "flash_attn":
        import flash_attn

        module = flash_attn
    else:
        raise ValueError("Invalid framework. Use 'torch' or 'paddle'.")

    # 解析API路径
    parts = api_name.split(".")[1:]
    api_obj = module
    try:
        for part in parts:
            api_obj = getattr(api_obj, part)
    except Exception as e:
        print(f"Failed to parse API path for {api_name}: {e}")
        return ""
    # 尝试获取函数签名
    try:
        signature = inspect.signature(api_obj)
        params = []
        for name, param in signature.parameters.items():
            if param.default is inspect.Parameter.empty:
                params.append(name)
            else:
                # 使用 repr 保证默认值可读且合法
                params.append(f"{name}={param.default!r}")
        return f"{api_name}({', '.join(params)})"
    except (ValueError, TypeError) as e:
        # inspect.signature 无法处理某些 built-in 或 C 扩展函数时会抛出异常
        print(f"Failed to inspect signature for {api_name}: {e}")
        return ""


def find_test_file(torch_api: str, paconvert_dir: str) -> str:
    """
    在paconvert的tests目录中递归查找测试文件

    Args:
        torch_api: Torch API名称
        paconvert_dir: paconvert目录路径

    Returns:
        测试文件路径
    """
    # 生成测试文件名（如：test_nn_functional_elu.py）
    parts = torch_api.split(".")[1:]
    file_name = "test_" + "_".join(parts) + ".py"

    # 在tests目录中递归查找
    for root, _, files in os.walk(os.path.join(paconvert_dir, "tests")):
        if file_name in files:
            return os.path.join(root, file_name)
    return ""


def extract_test_case_code(test_file: str) -> str:
    """
    从测试文件中提取第一个测试用例的代码

    Args:
        test_file: 测试文件路径

    Returns:
        测试用例代码字符串
    """
    with open(test_file, "r") as f:
        content = f.read()

    # 解析AST找到第一个测试函数
    module = ast.parse(content)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and isinstance(item.targets[0], ast.Name)
                    and item.targets[0].id == "pytorch_code"
                ):
                    return ast.get_source_segment(content, item)
    return ""


def get_torch_example(torch_api: str, paconvert_dir: str) -> str:
    """
    获取Torch API的示例代码

    Args:
        torch_api: Torch API名称
        paconvert_dir: paconvert目录路径

    Returns:
        Torch API调用代码
    """
    # 1. 查找测试文件
    test_file = find_test_file(torch_api, paconvert_dir)
    if not test_file:
        print(f"Warning: Test file not found for {torch_api}")
        return f"x.{torch_api.split('.')[-1]}()"

    # 2. 提取测试用例代码
    test_code = extract_test_case_code(test_file)
    if not test_code:
        print(f"Warning: Test case not found in {test_file}")
        return f"x.{torch_api.split('.')[-1]}()"

    # 3. 写入临时文件
    torch_temp_file = f"temp_generate_api_difference_{torch_api.replace('.', '_')}_torch_code.py"
    with open(torch_temp_file, "w") as f:
        f.write(test_code)

    # 4. 使用paconvert补全代码
    complete_file = f"temp_generate_api_difference_{torch_api.replace('.', '_')}_torch_code_complete.py"
    try:
        subprocess.run(
            [
                "python3.10",
                os.path.join(paconvert_dir, "paconvert", "main.py"),
                "-i",
                torch_temp_file,
                "-o",
                complete_file,
                "--only_complete",
            ],
            check=True,
            capture_output=True,
        )

        # 5. 替换import paddle为import torch
        with open(complete_file, "r") as f:
            lines = f.readlines()

        if lines and "import paddle" in lines[0]:
            lines[0] = "import torch\n"

        with open(complete_file, "w") as f:
            f.writelines(lines)

        # 6. 查找包含API调用的行
        with open(complete_file, "r") as f:
            for line in f:
                if torch_api.split(".")[-1] in line:
                    return line.strip()

    except subprocess.CalledProcessError as e:
        print(f"Warning: paconvert failed for {torch_api}: {e.stderr.decode()}")

    return f"x.{torch_api.split('.')[-1]}()"


def get_conversion_example(
    torch_code: str, torch_api: str, paconvert_dir: str
) -> str:
    """
    使用paconvert转换Torch代码为Paddle代码

    Args:
        torch_code: Torch代码
        torch_api: Torch API名称（用于临时文件命名）
        paconvert_dir: paconvert目录路径

    Returns:
        Paddle API调用代码
    """
    # 1. 创建临时文件
    temp_file = f"temp_generate_api_difference_{torch_api.replace('.', '_')}_torch_code_complete.py"
    with open(temp_file, "w") as f:
        f.write(torch_code)

    # 2. 运行paconvert转换
    paddle_file = f"temp_generate_api_difference_{torch_api.replace('.', '_')}_paddle_code.py"
    try:
        subprocess.run(
            [
                "python3.10",
                os.path.join(paconvert_dir, "paconvert", "main.py"),
                "-i",
                temp_file,
                "-o",
                paddle_file,
            ],
            check=True,
            capture_output=True,
        )

        # 3. 查找包含API调用的行
        with open(paddle_file, "r") as f:
            for line in f:
                if "paddle." in line and torch_api.split(".")[-1] in line:
                    return line.strip()

    except subprocess.CalledProcessError as e:
        print(
            f"Warning: paconvert conversion failed for {torch_api}: {e.stderr.decode()}"
        )

    return ""


def generate_invok_diff_only_docs(
    output_dir: str, paconvert_dir: str, overwrite: bool = False
):
    """
    生成"仅 API 调用方式不一致"类别的API差异文档

    Args:
        api_mapping_path: api_mapping.json的路径
        output_dir: 输出目录
        overwrite: 是否覆盖已有文件
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    api_mapping_path = os.path.join(script_dir, "api_mapping.json")
    attribute_mapping_path = os.path.join(script_dir, "attribute_mapping.json")

    output_dir = os.path.join(script_dir, output_dir)

    whitelist_api = [
        "fairscale.nn.model_parallel.initialize.get_model_parallel_rank",
        "fairscale.nn.model_parallel.initialize.get_model_parallel_world_size",
        "flash_attn.__version__.split",
    ]

    # 读取API映射
    api_mapping = load_mapping_json(api_mapping_path)
    attribute_mapping = load_mapping_json(attribute_mapping_path)
    api_mapping = api_mapping | attribute_mapping

    # 定义属于invok_diff_only类别的Matcher
    invok_diff_matchers = [
        "ChangeAPIMatcher",
        "TensorFunc2PaddleFunc",
        "Func2Attribute",
        "Attribute2Func",
        "NumelMatcher",
        "Is_InferenceMatcher",
    ]

    # 创建输出目录
    output_dir += "/invok_diff_only"
    os.makedirs(output_dir, exist_ok=True)

    # 生成文档
    for torch_api, mapping in api_mapping.items():
        if (
            mapping.get("Matcher") in invok_diff_matchers
            and torch_api not in whitelist_api
        ):
            paddle_api = mapping["paddle_api"]
            print(f"Processing: {torch_api} -> {paddle_api}")
            # 生成文件名
            file_name = f"{torch_api}.md"
            file_path = os.path.join(output_dir, file_name)

            # 检查是否覆盖
            # if os.path.exists(file_path) and not overwrite:
            #    print(f"Skipping existing file: {file_path}")
            #    continue

            # 获取URL
            torch_url = get_pytorch_url(torch_api)
            paddle_url = get_paddle_url(paddle_api)

            # 获取函数签名
            module = torch_api.split(".")[0]
            torch_signature = get_function_signature(torch_api, module)
            paddle_signature = get_function_signature(paddle_api, "paddle")

            if torch_signature == "" or paddle_signature == "":
                continue

            # 生成转写示例
            # 从torch_api中提取函数名
            torch_func_name = torch_api.split(".")[-1]
            torch_example = get_torch_example(torch_api, paconvert_dir)
            paddle_example = get_conversion_example(
                torch_example, torch_api, paconvert_dir
            )

            # 生成文档内容
            content = f"## [ 仅 API 调用方式不一致 ]{torch_api}\n\n"
            content += f"### [{torch_api}]({torch_url})\n\n"
            content += "```python\n"
            content += f"{torch_signature}\n"
            content += "```\n\n"
            content += f"### [{paddle_api}]({paddle_url})\n\n"
            content += "```python\n"
            content += f"{paddle_signature}\n"
            content += "```\n\n"
            content += "两者功能一致，但调用方式不一致，具体如下：\n\n"
            content += "### 转写示例\n\n"
            content += "```python\n"
            content += f"# PyTorch 写法:\n{torch_example}\n\n"
            content += f"# Paddle 写法:\n{paddle_example}\n"
            content += "```\n"

            # 保存文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"Generated: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate API difference documents for invok_diff_only category"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test",
        help="Output directory for generated docs",
    )
    parser.add_argument(
        "--paconvert_dir",
        type=str,
        default="/workspace/PaConvert",
        help="Path to PaConvert directory",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )

    args = parser.parse_args()

    # 生成文档
    generate_invok_diff_only_docs(
        output_dir=args.output_dir,
        paconvert_dir=args.paconvert_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
