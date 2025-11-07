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


def get_function_signature(api_name: str, framework: str) -> str:
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
        print(f"WARNING: Invalid framework {framework}", file=sys.stderr)
        return ""

    parts = api_name.split(".")[1:]
    api_obj = module
    try:
        for part in parts:
            api_obj = getattr(api_obj, part)
    except Exception as e:
        print(
            f"WARNING: Failed to parse API path {api_name}: {e}",
            file=sys.stderr,
        )
        return ""

    try:
        if inspect.isclass(api_obj):
            sig = inspect.signature(api_obj.__init__)
            params = []
            skip_self = True
            for name, param in sig.parameters.items():
                if skip_self and name == "self":
                    skip_self = False
                    continue
                params.append(format_param(param))
            return f"{api_name}({', '.join(params)})"
        else:
            sig = inspect.signature(api_obj)
            params = [format_param(param) for param in sig.parameters.values()]
            return f"{api_name}({', '.join(params)})"
    except (ValueError, TypeError):
        return parse_signature_from_help(api_obj, api_name)


def format_param(param: inspect.Parameter) -> str:
    if param.kind == param.VAR_POSITIONAL:
        return f"*{param.name}"
    elif param.kind == param.VAR_KEYWORD:
        return f"**{param.name}"
    else:
        return param.name


def parse_signature_from_help(api_obj, api_name: str) -> str:
    help_output = get_help_output(api_obj, api_name)
    if not help_output:
        return f"{api_name}(...)"

    signature_line = extract_signature_line(help_output, api_name)
    if signature_line:
        return normalize_signature(signature_line, api_name)

    return f"{api_name}()"


def get_help_output(api_obj, api_name: str) -> str | None:
    try:
        with io.StringIO() as buffer:
            with contextlib.redirect_stdout(buffer):
                help(api_obj)
            return buffer.getvalue()
    except Exception as e:
        print(
            f"WARNING: Failed to get help output for {api_name}: {e}",
            file=sys.stderr,
        )
        return ""


def extract_signature_line(help_text: str, api_name: str) -> str | None:
    lines = help_text.split("\n")
    base_name = api_name.split(".")[-1]

    patterns = [
        rf"^{re.escape(base_name)}\s*\([^)]*\)",
        rf"^\s*[\| ]*\w+\s+{re.escape(base_name)}\s*\([^)]*\)",
        rf"^class\s+\w+\(.*\):\s*\n\s*[\| ]*{re.escape(base_name)}\s*\([^)]*\)",
    ]

    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.search(pattern, line.strip()):
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

    print(
        f"WARNING: Failed to extract signature line for {api_name}",
        file=sys.stderr,
    )
    return ""


def normalize_signature(signature_line: str, api_name: str) -> str:
    match = re.search(r"\(([^)]*)\)", signature_line)
    if not match:
        return f"{api_name}()"

    params_str = match.group(1)
    if params_str == "...":
        return f"{api_name}()"

    params = []
    current_param = []
    depth = 0

    for char in params_str + ",":
        if char == "," and depth == 0:
            param = "".join(current_param).strip()
            if param:
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

    final_params = []
    for param in params:
        if param.startswith("*") and param.endswith(": Any"):
            param = param.replace(": Any", "")
        elif "=" in param and not param.startswith("*"):
            param = param.split("=")[0].strip()
        final_params.append(param)

    return f"{api_name}({', '.join(final_params)})"


def extract_param_name(param_str: str) -> str | None:
    param_str = param_str.strip()
    if not param_str:
        return None

    if param_str.startswith("*") and param_str.endswith(": Any"):
        return param_str.replace(": Any", "")

    if ":" in param_str:
        colon_pos = param_str.find(":")
        param_name = param_str[:colon_pos].strip()
        return param_name
    else:
        return (
            param_str.split("=")[0].strip() if "=" in param_str else param_str
        )


def find_test_file(torch_api: str, paconvert_dir: str) -> str:
    head = torch_api.split(".")[0]
    if head == "flash_attn":
        parts = torch_api.split(".")
    elif torch_api.startswith("torch.distributed"):
        parts = torch_api.split(".")[2:]
    else:
        parts = torch_api.split(".")[1:]
    file_name = "test_" + "_".join(parts) + ".py"

    if head == "torchvision":
        file_name = "test_" + torch_api.split(".")[-1] + ".py"

    test_dir = os.path.join(paconvert_dir, "tests")
    for root, _, files in os.walk(test_dir):
        if file_name in files:
            return os.path.join(root, file_name)

    print(f"WARNING: Test file not found for {torch_api}", file=sys.stderr)
    return ""


def extract_test_case_code(test_file: str) -> str:
    if not test_file:
        return ""
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            content = f.read()

        module = ast.parse(content)
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith(
                "test_"
            ):
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "pytorch_code"
                            ):
                                if isinstance(item.value, ast.Call):
                                    call_func = item.value.func
                                    if (
                                        isinstance(call_func, ast.Attribute)
                                        and isinstance(
                                            call_func.value, ast.Name
                                        )
                                        and call_func.value.id == "textwrap"
                                        and call_func.attr == "dedent"
                                    ):
                                        arg = item.value.args[0]
                                        if isinstance(arg, ast.Str):
                                            return textwrap.dedent(arg.s)
                                        elif isinstance(
                                            arg, ast.Constant
                                        ) and isinstance(arg.value, str):
                                            return textwrap.dedent(arg.value)
        # 如果没找到 pytorch_code，直接返回整个函数体作为 fallback
        print(
            f"WARNING: pytorch_code not found in {test_file}, returning full test function body",
            file=sys.stderr,
        )
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith(
                "test_"
            ):
                return ast.unparse(node)
        return ""
    except Exception as e:
        print(
            f"WARNING: Failed to extract test case code from {test_file}: {e}",
            file=sys.stderr,
        )
        return ""


def get_torch_example(torch_api: str, paconvert_dir: str) -> str:
    test_file = find_test_file(torch_api, paconvert_dir)
    if not test_file:
        return ""

    test_code = extract_test_case_code(test_file)
    if not test_code:
        return ""

    torch_temp_file = f"temp_generate_api_difference_{torch_api.replace('.', '_')}_torch_code.py"
    with open(torch_temp_file, "w", encoding="utf-8") as f:
        f.write(test_code)

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

        with open(complete_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith("import paddle"):
                lines[i] = "import torch\n"
        # if lines and "import paddle" in lines[0]:
        #     lines[0] = "import torch\n"

        with open(complete_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        with open(complete_file, "r", encoding="utf-8") as f:
            return "".join(lines)

        return ""

    except subprocess.CalledProcessError as e:
        print(
            f"WARNING: paconvert failed for {torch_api}: {e.stderr.decode().strip()}",
            file=sys.stderr,
        )
        return ""
    except Exception as e:
        print(
            f"WARNING: Unexpected error in get_torch_example for {torch_api}: {e}",
            file=sys.stderr,
        )
        return ""


def get_conversion_example(
    torch_code: str, torch_api: str, paddle_api: str, paconvert_dir: str
) -> str:
    if not torch_code:
        return ""

    temp_file = f"temp_generate_api_difference_{torch_api.replace('.', '_')}_torch_code_complete.py"

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

        with open(paddle_file, "r", encoding="utf-8") as f:
            return "".join(f.readlines())

        return ""

    except subprocess.CalledProcessError as e:
        print(
            f"WARNING: paconvert conversion failed for {torch_api}: {e.stderr.decode().strip()}",
            file=sys.stderr,
        )
        return ""
    except Exception as e:
        print(
            f"WARNING: Unexpected error in get_conversion_example for {torch_api}: {e}",
            file=sys.stderr,
        )
        return ""


def generate_invok_diff_only_docs(
    output_dir: str, paconvert_dir: str, overwrite: bool = False
):
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

    api_mapping = load_mapping_json(api_mapping_path)
    attribute_mapping = load_mapping_json(attribute_mapping_path)
    api_mapping = api_mapping | attribute_mapping

    invok_diff_matchers = [
        "ChangeAPIMatcher",
        "NumelMatcher",
        "Is_InferenceMatcher",
    ]

    special_matchers = [
        "TensorFunc2PaddleFunc",
        "Func2Attribute",
        "Attribute2Func",
    ]

    test_output_dir = os.path.join(output_dir, "invok_diff_only")
    actually_output_dir = os.path.join(
        output_dir, "../../api_difference/invok_diff_only"
    )
    os.makedirs(test_output_dir, exist_ok=True)

    total = 0

    for torch_api, mapping_info in api_mapping.items():
        if torch_api in whitelist_api or torch_api in no_need_list:
            continue
        matcher = mapping_info.get("Matcher", "")
        valid = False
        # ChangeAPIMatcher、TensorFunc2PaddleFunc、Func2Attribute、Attribute2Func类别
        if matcher in special_matchers:
            has_unsupport_args = "unsupport_args" in mapping_info
            has_kwargs_change = "kwargs_change" in mapping_info
            has_paddle_default_kwargs = "paddle_default_kwargs" in mapping_info
            if has_unsupport_args:
                print(
                    f"[torch_more_args] {torch_api} -> {mapping_info.get('paddle_api', 'N/A')}"
                )
                continue
            elif has_kwargs_change:
                print(
                    f"[args_name_diff] {torch_api} -> {mapping_info.get('paddle_api', 'N/A')}"
                )
                continue
            elif has_paddle_default_kwargs:
                print(
                    f"[paddle_more_args_or_default_diff] {torch_api} -> {mapping_info.get('paddle_api', 'N/A')}"
                )
                continue
            valid = True

        if matcher in invok_diff_matchers or valid:
            paddle_api = mapping_info["paddle_api"]
            if torch_api.startswith("torch"):
                continue
            print(f"Processing: {torch_api} -> {paddle_api}")
            file_name = f"{torch_api}.md"
            file_path = os.path.join(test_output_dir, file_name)
            if overwrite:
                file_path = os.path.join(actually_output_dir, file_name)

            try:
                torch_url = get_pytorch_url(torch_api)
                paddle_url = get_paddle_url(paddle_api)

                module = torch_api.split(".")[0]
                torch_signature = get_function_signature(torch_api, module)
                paddle_signature = get_function_signature(paddle_api, "paddle")

                torch_example = get_torch_example(torch_api, paconvert_dir)
                paddle_example = get_conversion_example(
                    torch_example, torch_api, paddle_api, paconvert_dir
                )
                if not torch_signature:
                    torch_signature = ""
                if not paddle_signature:
                    paddle_signature = ""
                if not torch_example:
                    torch_example = ""
                if not paddle_example:
                    paddle_example = ""

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

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    total += 1

                print(f"Generated: {file_path}")

            except Exception as e:
                print(
                    f"WARNING: Unexpected error processing {torch_api}: {e}",
                    file=sys.stderr,
                )
                continue
    print("total: ", total)


def main():
    parser = argparse.ArgumentParser(
        description="Generate API difference documents for invok_diff_only category"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_new",
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

    generate_invok_diff_only_docs(
        output_dir=args.output_dir,
        paconvert_dir=args.paconvert_dir,
        overwrite=args.overwrite,
    )

    os.system("rm -rf temp_generate_api_difference_*")


if __name__ == "__main__":
    main()
