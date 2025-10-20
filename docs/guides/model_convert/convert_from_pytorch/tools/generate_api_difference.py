from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import io
import os
import re
import subprocess
import sys
import textwrap

from utils import (
    extract_no_need_convert_list,
    get_paddle_url,
    get_pytorch_url,
    load_mapping_json,
)


class APIConversionError(Exception):
    """自定义异常类，用于API转换过程中的错误"""

    def __init__(self, message, api_name=None):
        self.message = message
        self.api_name = api_name
        super().__init__(self.message)

    def __str__(self):
        return f"API Conversion Error for {self.api_name}: {self.message}"


def get_function_signature(api_name: str, framework: str) -> str:
    """
    获取API的函数签名，支持普通函数、类方法、内置方法等

    Args:
        api_name: API的全限定名
        framework: 框架名称

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
        raise ValueError("Invalid framework.")

    # 解析API路径
    parts = api_name.split(".")[1:]
    api_obj = module

    try:
        for part in parts:
            api_obj = getattr(api_obj, part)
    except Exception as e:
        raise ValueError(f"Failed to parse API path: {e}") from e

    # 优先尝试inspect.signature（适用于普通函数和方法）
    try:
        # 处理类的情况（获取__init__方法）
        if inspect.isclass(api_obj):
            sig = inspect.signature(api_obj.__init__)
            # 移除self参数
            params = []
            first_param = True
            for name, param in sig.parameters.items():
                if first_param and name == "self":
                    first_param = False
                    continue
                first_param = False
                params.append(format_param(param))

            return f"{api_name}({', '.join(params)})"

        # 处理普通函数和方法
        sig = inspect.signature(api_obj)
        params = [format_param(param) for param in sig.parameters.values()]
        return f"{api_name}({', '.join(params)})"

    except (ValueError, TypeError):
        # 如果inspect失败，使用help信息解析
        return parse_signature_from_help(api_obj, api_name)


def format_param(param: inspect.Parameter) -> str:
    """格式化参数，保留*和**符号"""
    if param.kind == param.VAR_POSITIONAL:
        return f"*{param.name}"
    elif param.kind == param.VAR_KEYWORD:
        return f"**{param.name}"
    else:
        return param.name


def parse_signature_from_help(api_obj, api_name: str) -> str:
    """从help信息中解析函数签名"""

    # 捕获help输出
    help_output = get_help_output(api_obj, api_name)
    if not help_output:
        return f"{api_name}(...)"

    # 解析函数签名行
    signature_line = extract_signature_line(help_output, api_name)
    if signature_line:
        return normalize_signature(signature_line, api_name)

    return f"{api_name}(...)"


def get_help_output(api_obj, api_name: str) -> str | None:
    """获取API对象的help输出"""
    try:
        with io.StringIO() as buffer:
            with contextlib.redirect_stdout(buffer):
                help(api_obj)
            return buffer.getvalue()
    except:
        raise APIConversionError(f"Failed to get help output for {api_name}")


def extract_signature_line(help_text: str, api_name: str) -> str | None:
    """从help文本中提取签名行"""
    lines = help_text.split("\n")
    base_name = api_name.split(".")[-1]

    # 匹配模式：函数名后跟括号
    patterns = [
        # 匹配: numel(...) 或 forward(*args: Any, **kwargs: Any) -> Any
        rf"^{re.escape(base_name)}\s*\([^)]*\)",
        # 匹配: |  ZeroPad2d(padding: Union[int, tuple[int, int, int, int]]) -> None
        rf"^\s*[\| ]*\w+\s+{re.escape(base_name)}\s*\([^)]*\)",
        # 匹配类定义中的签名
        rf"^class\s+\w+\(.*\):\s*\n\s*[\| ]*{re.escape(base_name)}\s*\([^)]*\)",
    ]

    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.search(pattern, line.strip()):
                # 对于多行签名，合并后续行直到遇到空行或缩进减少
                signature = line.strip()
                j = i + 1
                while (
                    j < len(lines)
                    and lines[j].strip()
                    and not lines[j].strip().startswith(("def ", "class "))
                ):
                    signature += " " + lines[j].strip()
                    j += 1
                return signature

    raise APIConversionError(f"Failed to extract signature line for {api_name}")


def normalize_signature(signature_line: str, api_name: str) -> str:
    """规范化签名格式"""
    # 提取括号内的内容
    match = re.search(r"\(([^)]*)\)", signature_line)
    if not match:
        return f"{api_name}()"

    params_str = match.group(1)

    # 处理不同的参数格式
    if params_str == "...":  # 如: numel(...)
        return f"{api_name}()"

    # 解析参数，移除类型注解和默认值
    params = []
    current_param = []
    depth = 0  # 处理嵌套括号

    for char in params_str + ",":  # 添加逗号确保处理最后一个参数
        if char == "," and depth == 0:
            param = "".join(current_param).strip()
            if param:
                # 提取参数名（移除类型注解）
                param_name = extract_param_name(param)
                if param_name:
                    params.append(param_name)
            current_param = []
        else:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            current_param.append(char)

    # 处理特殊参数格式
    final_params = []
    for param in params:
        if param.startswith("*") and param.endswith(": Any"):
            # 处理 *args: Any, **kwargs: Any
            param = param.replace(": Any", "")
        elif "=" in param and not param.startswith("*"):
            # 移除默认值，保留参数名
            param = param.split("=")[0].strip()

        final_params.append(param)

    return f"{api_name}({', '.join(final_params)})"


def extract_param_name(param_str: str) -> str | None:
    """从参数字符串中提取参数名"""
    param_str = param_str.strip()

    if not param_str:
        return None

    # 处理可变参数 *args, **kwargs
    if param_str.startswith("*") and param_str.endswith(": Any"):
        return param_str.replace(": Any", "")

    # 移除类型注解
    if ":" in param_str:
        # 找到第一个冒号（类型注解开始）
        colon_pos = param_str.find(":")
        param_name = param_str[:colon_pos].strip()

        # 检查是否是可变参数
        if param_name.startswith("*"):
            return param_name
        else:
            return param_name
    else:
        # 没有类型注解，直接返回
        return (
            param_str.split("=")[0].strip() if "=" in param_str else param_str
        )


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
    head = torch_api.split(".")[0]
    if head == "flash_attn":
        parts = torch_api.split(".")
    else:
        parts = torch_api.split(".")[1:]
    file_name = "test_" + "_".join(parts) + ".py"

    # 在tests目录中递归查找
    test_dir = os.path.join(paconvert_dir, "tests")
    for root, _, files in os.walk(test_dir):
        if file_name in files:
            return os.path.join(root, file_name)

    raise APIConversionError(f"Test file not found for {torch_api}", torch_api)


def extract_test_case_code(test_file: str) -> str:
    """
    从测试文件中提取第一个测试用例的pytorch代码

    Args:
        test_file: 测试文件路径

    Returns:
        提取的pytorch代码字符串
    """
    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析AST找到第一个测试函数
    module = ast.parse(content)

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # 在测试函数中查找pytorch_code的赋值语句
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "pytorch_code"
                        ):
                            # 检查赋值语句的值是否是函数调用
                            if isinstance(item.value, ast.Call):
                                call_func = item.value.func
                                # 检查是否是textwrap.dedent调用
                                if (
                                    isinstance(call_func, ast.Attribute)
                                    and isinstance(call_func.value, ast.Name)
                                    and call_func.value.id == "textwrap"
                                    and call_func.attr == "dedent"
                                ):
                                    # 获取dedent的第一个参数（应该是字符串）
                                    if item.value.args and isinstance(
                                        item.value.args[0], ast.Str
                                    ):
                                        # 返回字符串内容，去除首尾的三引号
                                        code_str = item.value.args[0].s
                                        return textwrap.dedent(code_str)

                                    # 处理Python 3.8+的ast.Constant节点
                                    elif (
                                        item.value.args
                                        and isinstance(
                                            item.value.args[0], ast.Constant
                                        )
                                        and isinstance(
                                            item.value.args[0].value, str
                                        )
                                    ):
                                        code_str = item.value.args[0].value
                                        return textwrap.dedent(code_str)

    raise APIConversionError(f"Test case not found in {test_file}", test_file)


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

    # 2. 提取测试用例代码
    test_code = extract_test_case_code(test_file)

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
        raise APIConversionError(
            f"paconvert failed: {e.stderr.decode().strip()}", torch_api
        ) from e

    # 如果执行到这里，说明没有找到API调用
    raise APIConversionError(
        f"torch API call not found in converted code for {torch_api}", torch_api
    )


def get_conversion_example(
    torch_code: str, torch_api: str, paddle_api: str, paconvert_dir: str
) -> str:
    """
    使用paconvert转换Torch代码为Paddle代码

    Args:
        torch_code: Torch代码
        torch_api: Torch API名称（用于临时文件命名）
        paddle_api: Paddle API名称
        paconvert_dir: paconvert目录路径

    Returns:
        Paddle API调用代码
    """
    # 1. 创建临时文件
    temp_file = f"temp_generate_api_difference_{torch_api.replace('.', '_')}_torch_code_complete.py"
    # with open(temp_file, "w") as f:
    #     f.write(torch_code)

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
                if paddle_api.split(".")[-1] in line:
                    return line.strip()

    except subprocess.CalledProcessError as e:
        raise APIConversionError(
            f"paconvert conversion failed: {e.stderr.decode().strip()}",
            torch_api,
        ) from e

    # 如果执行到这里，说明没有找到API调用
    raise APIConversionError(
        f"paddle API call not found in converted code for {torch_api}",
        torch_api,
    )


def generate_invok_diff_only_docs(
    output_dir: str, paconvert_dir: str, overwrite: bool = False
):
    """
    生成"仅 API 调用方式不一致"类别的API差异文档

    Args:
        output_dir: 输出目录
        paconvert_dir: PaConvert目录路径
        overwrite: 是否覆盖已有文件
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    api_mapping_path = os.path.join(script_dir, "api_mapping.json")
    attribute_mapping_path = os.path.join(script_dir, "attribute_mapping.json")
    global_var_path = os.path.join(script_dir, "global_var.py")
    no_need_list = extract_no_need_convert_list(global_var_path)

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
    test_output_dir = os.path.join(output_dir, "invok_diff_only")
    actually_output_dir = os.path.join(
        output_dir, "../../api_difference/invok_diff_only"
    )
    os.makedirs(test_output_dir, exist_ok=True)

    # 生成文档
    for torch_api, mapping in api_mapping.items():
        if (
            mapping.get("Matcher") in invok_diff_matchers
            and torch_api not in whitelist_api
            and torch_api not in no_need_list
        ):
            paddle_api = mapping["paddle_api"]
            print(f"Processing: {torch_api} -> {paddle_api}")
            # 生成文件名
            file_name = f"{torch_api}.md"
            file_path = os.path.join(test_output_dir, file_name)
            if overwrite:
                file_path = os.path.join(actually_output_dir, file_name)

            try:
                # 获取URL
                torch_url = get_pytorch_url(torch_api)
                paddle_url = get_paddle_url(paddle_api)

                # 获取函数签名
                module = torch_api.split(".")[0]
                torch_signature = get_function_signature(torch_api, module)
                paddle_signature = get_function_signature(paddle_api, "paddle")

                # 生成转写示例
                torch_example = get_torch_example(torch_api, paconvert_dir)
                paddle_example = get_conversion_example(
                    torch_example, torch_api, paddle_api, paconvert_dir
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
                content += f"# PyTorch 写法\n{torch_example}\n\n"
                content += f"# Paddle 写法\n{paddle_example}\n"
                content += "```\n"

                # 保存文件
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                print(f"Generated: {file_path}")

            except APIConversionError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"UNEXPECTED ERROR: {e} for {torch_api}", file=sys.stderr)
                continue


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
