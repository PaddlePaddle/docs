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
        raise ValueError(f"Invalid framework: {framework}")

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
            for name, param in sig.parameters.items():
                if name == "self" or param.name == "self":
                    continue
                params.append(format_param(param))
            return f"{api_name}({', '.join(params)})"

        # 处理普通函数和方法
        sig = inspect.signature(api_obj)
        params = []
        for name, param in sig.parameters.values():
            if name == "self" or param.name == "self":
                continue
            params.append(format_param(param))
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
    help_output = get_help_output(api_obj, api_name)
    if not help_output:
        raise APIConversionError(f"Failed to get help output for {api_name}")

    # 尝试从help输出中提取签名行
    signature_line = extract_signature_line(help_output, api_name)
    if not signature_line:
        raise APIConversionError(
            f"Failed to extract signature line for {api_name}"
        )
    signature = normalize_signature(signature_line, api_name)
    print("LOGGING:", api_name, help_output[:5], signature)
    return signature


def get_help_output(api_obj, api_name: str) -> str | None:
    """获取API对象的help输出"""
    try:
        with io.StringIO() as buffer:
            with contextlib.redirect_stdout(buffer):
                help(api_obj)
            return buffer.getvalue()
    except Exception as e:
        raise APIConversionError(
            f"Failed to get help output for {api_name}: {e!s}"
        )


def extract_signature_line(help_text: str, api_name: str) -> str | None:
    """从help文本中提取签名行"""
    lines = help_text.split("\n")
    base_name = api_name.split(".")[-1]
    signature_line = None

    # 匹配模式：函数名后跟括号
    patterns = [
        # 匹配: numel(input: Tensor) -> int
        rf"^{re.escape(base_name)}\s*\([^)]*\)",
        # 匹配: |  ZeroPad2d(padding: Union[int, tuple[int, int, int, int]]) -> None
        rf"^\s*[\| ]*\w+\s+{re.escape(base_name)}\s*\([^)]*\)",
        # 匹配类定义中的签名
        rf"^class\s+\w+\(.*\):\s*\n\s*[\| ]*{re.escape(base_name)}\s*\([^)]*\)",
    ]

    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.search(pattern, line.strip()):
                # 提取括号内的内容
                match = re.search(r"\(([^)]*)\)", line.strip())
                if not match:
                    continue
                params_str = match.group(1)

                # 跳过无效的签名（如numel(...)）
                if (
                    params_str.strip() == "..."
                    or params_str.strip() == "..." * 2
                ):
                    continue

                # 处理多行签名
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

    # 如果未找到有效签名，尝试匹配包含"->"的行
    for i, line in enumerate(lines):
        if " -> " in line and base_name in line:
            # 提取完整签名行
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

    raise APIConversionError(
        f"Failed to extract signature line for {api_name} from help"
    )


def normalize_signature(signature_line: str, api_name: str) -> str:
    """规范化签名格式，正确处理嵌套括号"""
    # 提取括号内的内容
    match = re.search(r"\(([^)]*)\)", signature_line)
    if not match:
        return f"{api_name}()"

    params_str = match.group(1)
    params = []

    # 处理参数列表（支持逗号分隔，正确处理嵌套括号）
    current_param = []
    stack = []  # 用于跟踪括号嵌套

    # 添加逗号确保处理最后一个参数
    for char in params_str + ",":
        # 处理左括号：圆括号、方括号、尖括号
        if char in "([{<":
            stack.append(char)
        # 处理右括号：匹配对应的左括号
        elif char in ")]}>":
            if stack:
                # 检查是否匹配（简化处理，不检查具体类型）
                stack.pop()

        # 当栈为空时，遇到逗号才分割参数
        if char == "," and not stack:
            param = "".join(current_param).strip()
            if param:
                params.append(extract_param_name(param))
            current_param = []
        else:
            current_param.append(char)

    # 移除无效参数
    params = [p for p in params if p]

    return f"{api_name}({', '.join(params)})"


def extract_param_name(param_str: str) -> str | None:
    """从参数字符串中提取参数名（移除类型注解和默认值）"""
    param_str = param_str.strip()

    if not param_str:
        return None
    if "self" == param_str:
        return None
    # 处理可变参数 *args, **kwargs
    if param_str.startswith("*") or param_str.startswith("**"):
        # 移除类型注解（如 *args: Any）
        if ": " in param_str:
            param_name = param_str.split(":")[0].strip()
            if "self" == param_name:
                return None
            return param_name
        return param_str

    # 移除类型注解（如 input: Tensor）
    if ":" in param_str:
        param_name = param_str.split(":")[0].strip()
        # # 移除默认值（如 input=None）
        # if "=" in param_name:
        #     param_name = param_name.split("=")[0].strip()
        #     if "self" == param_name:
        #         return None
        #     return param_name
        if "self" == param_name.split("=")[0].strip():
            return None
        return param_name

    # 移除默认值（如 input=None）
    # if "=" in param_str:
    #     return param_str.split("=")[0].strip()

    return param_str


def find_test_file(torch_api: str, paconvert_dir: str) -> str:
    """在paconvert的tests目录中递归查找测试文件"""
    # 生成测试文件名（如：test_nn_functional_elu.py）
    parts = torch_api.split(".")[1:]
    file_name = "test_" + "_".join(parts) + ".py"

    # 在tests目录中递归查找
    test_dir = os.path.join(paconvert_dir, "tests")
    for root, _, files in os.walk(test_dir):
        if file_name in files:
            return os.path.join(root, file_name)

    raise APIConversionError(f"Test file not found for {torch_api}", torch_api)


def extract_test_case_code(test_file: str) -> str:
    """从测试文件中提取第一个测试用例的pytorch代码"""
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
                            # 处理textwrap.dedent调用
                            if isinstance(item.value, ast.Call):
                                if (
                                    isinstance(item.value.func, ast.Attribute)
                                    and item.value.func.value.id == "textwrap"
                                    and item.value.func.attr == "dedent"
                                ):
                                    if item.value.args and isinstance(
                                        item.value.args[0], ast.Str
                                    ):
                                        return textwrap.dedent(
                                            item.value.args[0].s
                                        )
                                    elif (
                                        item.value.args
                                        and isinstance(
                                            item.value.args[0], ast.Constant
                                        )
                                        and isinstance(
                                            item.value.args[0].value, str
                                        )
                                    ):
                                        return textwrap.dedent(
                                            item.value.args[0].value
                                        )

    raise APIConversionError(f"Test case not found in {test_file}", test_file)


def get_torch_example(torch_api: str, paconvert_dir: str) -> str:
    """获取Torch API的示例代码"""
    # 1. 查找测试文件
    test_file = find_test_file(torch_api, paconvert_dir)

    # 2. 提取测试用例代码
    test_code = extract_test_case_code(test_file)

    # 3. 写入临时文件
    base_name = torch_api.replace(".", "_")
    temp_file = f"temp_{base_name}_torch_code.py"
    with open(temp_file, "w") as f:
        f.write(test_code)

    # 4. 使用paconvert补全代码
    complete_file = f"temp_{base_name}_torch_code_complete.py"
    try:
        subprocess.run(
            [
                "python3.10",
                os.path.join(paconvert_dir, "paconvert", "main.py"),
                "-i",
                temp_file,
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

        for i, line in enumerate(lines):
            if "import paddle" in line:
                lines[i] = "import torch\n"

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
    """使用paconvert转换Torch代码为Paddle代码"""
    # 1. 创建临时文件（已由调用方创建，这里直接使用）
    # 2. 运行paconvert转换
    base_name = torch_api.replace(".", "_")
    paddle_file = f"temp_{base_name}_paddle_code.py"
    try:
        subprocess.run(
            [
                "python3.10",
                os.path.join(paconvert_dir, "paconvert", "main.py"),
                "-i",
                "temp_"
                + base_name
                + "_torch_code_complete.py",  # 使用已创建的complete_file
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
    output_dir: str,
    paconvert_dir: str,
    delete_temp_file: bool = False,
    overwrite: bool = False,
):
    """生成"仅 API 调用方式不一致"类别的API差异文档"""
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
    api_mapping = {**api_mapping, **attribute_mapping}

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

    # 记录所有临时文件路径
    temp_files = []

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
            base_name = torch_api.replace(".", "_")
            file_name = f"{torch_api}.md"
            file_path = os.path.join(test_output_dir, file_name)
            if overwrite:
                file_path = os.path.join(actually_output_dir, file_name)

            try:
                # 获取URL
                torch_url = get_pytorch_url(torch_api)
                paddle_url = get_paddle_url(paddle_api)

                # 获取函数签名
                torch_signature = get_function_signature(
                    torch_api, torch_api.split(".")[0]
                )
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

                # 记录临时文件
                temp_files.append(f"temp_{base_name}_torch_code.py")
                temp_files.append(f"temp_{base_name}_torch_code_complete.py")
                temp_files.append(f"temp_{base_name}_paddle_code.py")

            except APIConversionError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"UNEXPECTED ERROR: {e} for {torch_api}", file=sys.stderr)
                continue

    # 删除临时文件（如果需要）
    if delete_temp_file:
        for file in temp_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"Deleted temporary file: {file}")
                except Exception as e:
                    print(f"Failed to delete {file}: {e}", file=sys.stderr)


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
    parser.add_argument(
        "--delete_temp_file",
        action="store_true",
        help="Delete temporary files after generation",
    )

    args = parser.parse_args()

    # 生成文档
    generate_invok_diff_only_docs(
        output_dir=args.output_dir,
        paconvert_dir=args.paconvert_dir,
        delete_temp_file=args.delete_temp_file,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
