from __future__ import annotations

import argparse
import os
import re
import sys
import traceback

from utils import get_url


class APIDifferenceValidator:
    """API差异文档校验器"""

    # 类别映射
    CATEGORY_MAP = {
        "invok_only_diff": "仅 API 调用方式不一致",
        "args_name_diff": "仅参数名不一致",
        "paddle_more_args": "paddle 参数更多",
        "args_default_value_diff": "参数默认值不一致",
        "torch_more_args": "torch 参数更多",
        "input_args_usage_diff": "输入参数用法不一致",
        "input_args_type_diff": "输入参数类型不一致",
        "output_args_type_diff": "返回参数类型不一致",
        "composite_implement": "组合替代实现",
    }

    # 需要转写校验的类别
    REWRITE_VALIDATION_CATEGORIES = [
        "input_args_usage_diff",
        "input_args_type_diff",
        "output_args_type_diff",
        "composite_implement",
    ]

    # 有效的备注结尾
    VALID_REMARKS = ["需要转写", "暂无转写方式", "可直接删除"]

    def __init__(self, api_difference_dir: str, fix_url: bool = False):
        self.api_difference_dir = api_difference_dir
        self.total_files = 0
        self.valid_files = 0
        self.errors = []
        self.fix_url = fix_url
        self.url_errors = []
        self.url_fixed_count = 0

    def parse_parameters_from_signature(self, signature_line: str) -> list[str]:
        """从函数签名字符串中解析参数列表

        例如: torch.nn.LazyBatchNorm1d(eps=1e-05, momentum=0.1, affine=True)
        返回: ['eps', 'momentum', 'affine']
        """
        # 提取括号内的内容
        match = re.search(r"\((.*?)\)$", signature_line, re.DOTALL)
        if not match:
            return []

        params_str = match.group(1).strip()
        if not params_str:
            return []

        params = []

        params = params_str.replace(" ", "").split(",")
        # 移除类型标注和默认值
        params = [param.split("=")[0].strip() for param in params]
        params = [param.split(":")[0].strip() for param in params]

        # 移除 *, /, self 和 cls
        params = [param for param in params if param not in ["*", "/"]]
        if params and params[0] in ["self", "cls"]:
            params = params[1:]

        return params

    def extract_api_name_from_line(self, line: str) -> str | None:
        """从markdown链接行中提取API名称"""
        match = re.search(r"### \[(.*?)\]", line)
        if match:
            return match.group(1).replace(r"\_", "_")
        return None

    def log_error_with_details(
        self,
        error_file: str,
        error_message: str,
        exception: Exception | None,
        current_line: str = "",
    ) -> None:
        """记录错误信息及其详细信息到文件"""
        try:
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(f"{error_message}\n")
                if exception:
                    f.write(f"异常类型: {type(exception).__name__}\n")
                    f.write(f"异常信息: {exception!s}\n")
                    f.write(f"异常追踪: {traceback.format_exc()}\n")
                if current_line:
                    f.write(f"当前行内容: {current_line}\n")
                f.write("-" * 80 + "\n")
        except Exception as e:
            print(f"写入错误文件失败: {e}")

    def validate_parameter_table_remarks(
        self,
        table_lines: list[str],
        category_dir: str,
        file_path: str,
        section_index: int,
        torch_params: list[str] | None = None,
        paddle_params: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """校验参数映射表格的备注列"""
        missing_white_list = {
            "out",
            "device",
            "dtype",
        }  # 允许缺少映射方式说明的参数名称
        errors = []
        requires_rewrite_params = set()

        # 从表格中提取参数映射信息
        pytorch_params_in_table = []
        paddle_params_in_table = []

        # 跳过表头行（前两行）
        for i, line in enumerate(table_lines[2:], start=2):
            if not line.strip() or not line.startswith("|"):
                continue

            # 解析表格行
            parts = [part.strip() for part in line.split("|") if part.strip()]
            if len(parts) < 3:
                continue

            pytorch_param, paddle_param, remark = parts[0], parts[1], parts[2]

            # 记录表格中的参数
            if pytorch_param != "-":
                pytorch_params_in_table.append(pytorch_param)
            if paddle_param != "-":
                # 处理多个 paddle 参数的情况（用逗号分隔）
                paddle_param_list = [p.strip() for p in paddle_param.split(",")]
                paddle_params_in_table.extend(paddle_param_list)
            else:
                paddle_param_list = []

            # 功能1: 检查参数名称是否出现在对应 API 的参数签名列表中
            # 跳过返回值相关的行(包含"返回值"关键词的任何参数名)
            is_pytorch_return = (
                pytorch_param == "-" or "返回值" in pytorch_param
            )

            if torch_params is not None and not is_pytorch_return:
                if pytorch_param not in torch_params and pytorch_param not in [
                    "self",
                    "cls",
                ]:
                    errors.append(
                        f"参数映射错误: {file_path} - PyTorch 参数 '{pytorch_param}' 不在 API 参数签名中 (签名参数: {torch_params})"
                    )

            # 检查 paddle_param 是否包含"返回值"字样（支持各种变体）
            is_return_value = paddle_param == "-" or "返回值" in paddle_param

            if paddle_params is not None and not is_return_value:
                for param in paddle_param_list:
                    # 同样检查每个参数是否包含"返回值"
                    if (
                        "返回值" not in param
                        and param not in paddle_params
                        and param not in ["self", "cls"]
                    ):
                        errors.append(
                            f"参数映射错误: {file_path} - Paddle 参数 '{param}' 不在 API 参数签名中 (签名参数: {paddle_params})"
                        )

            # 检查备注是否以句号结尾
            if remark and not remark.endswith("。"):
                errors.append(
                    f"备注未以句号结尾: {file_path} - 参数 '{pytorch_param}' 的备注: '{remark}'"
                )

            # 检查torch参数更多类别的特殊要求
            if category_dir == "torch_more_args" and (
                (pytorch_param != "-" and paddle_param == "-")
                or "需要转写" in remark
            ):
                if not any(
                    valid_remark in remark
                    for valid_remark in self.VALID_REMARKS
                ):
                    errors.append(
                        f"缺少转写说明: {file_path} - 参数 '{pytorch_param}' 的备注应包含{self.VALID_REMARKS}之一"
                    )

                if "需要转写" in remark:
                    requires_rewrite_params.add(pytorch_param)

            # 检查需要转写校验的类别
            if (
                category_dir in self.REWRITE_VALIDATION_CATEGORIES
                and "需要转写" in remark
            ):
                if pytorch_param != "-":
                    requires_rewrite_params.add(pytorch_param)

        # 功能2: 检查参数映射是否包含 torch 全部参数
        if torch_params is not None:
            missing_torch_params = set(torch_params) - set(
                pytorch_params_in_table
            )
            if missing_torch_params > missing_white_list:
                errors.append(
                    f"参数映射不完整: {file_path} - 缺少 PyTorch 参数: {sorted(missing_torch_params)}"
                )

        # 功能3: 检查参数映射的顺序是否与签名中的顺序一致（torch 参数优先级最高）
        if torch_params is not None and len(pytorch_params_in_table) > 0:
            torch_params_order = [
                p for p in pytorch_params_in_table if p in torch_params
            ]

            expected_order = [
                p for p in torch_params if p in torch_params_order
            ]

            if torch_params_order != expected_order:
                errors.append(
                    f"参数顺序错误: {file_path} - PyTorch 参数顺序不符合签名顺序\n"
                    f"  当前顺序: {torch_params_order}\n"
                    f"  应为顺序: {expected_order}"
                )

        return requires_rewrite_params, errors

    def validate_rewrite_example(
        self,
        section_lines: list[str],
        current_index: int,
        requires_rewrite_params: set,
        category_dir: str,
        file_path: str,
        section_index: int,
    ) -> tuple[bool, list[str]]:
        """校验转写示例部分"""
        errors = []

        if category_dir == "invok_only_diff":
            return self._validate_required_rewrite_example(
                section_lines, current_index, file_path, section_index
            )
        elif category_dir in [
            *self.REWRITE_VALIDATION_CATEGORIES,
            "torch_more_args",
            "paddle_more_args",
            "args_default_value_diff",
            "composite_implement",
        ]:
            return self._validate_conditional_rewrite_example(
                section_lines,
                current_index,
                requires_rewrite_params,
                category_dir,
                file_path,
                section_index,
            )
        else:  # args_name_diff 类别不应包含转写示例
            if (
                current_index < len(section_lines)
                and section_lines[current_index] == "### 转写示例"
            ):
                errors.append(
                    f"文件格式错误: {file_path} - 此类别不应包含'### 转写示例'"
                )
                return False, errors
            return True, errors

    def _validate_required_rewrite_example(
        self,
        section_lines: list[str],
        current_index: int,
        file_path: str,
        section_index: int,
    ) -> tuple[bool, list[str]]:
        """校验必须包含的转写示例"""
        errors = []

        if (
            current_index >= len(section_lines)
            or section_lines[current_index] != "### 转写示例"
        ):
            current_line = (
                section_lines[current_index]
                if current_index < len(section_lines)
                else "EOF"
            )
            errors.append(
                f"文件格式错误: {file_path} - 缺少'### 转写示例'，当前行: '{current_line}'"
            )
            return False, errors

        current_index += 1

        if (
            current_index >= len(section_lines)
            or section_lines[current_index] != "```python"
        ):
            current_line = (
                section_lines[current_index]
                if current_index < len(section_lines)
                else "EOF"
            )
            errors.append(
                f"文件格式错误: {file_path} - 转写示例代码块开始错误，当前行: '{current_line}'"
            )
            return False, errors

        # 查找代码块结束
        example_code_end_found = False
        while current_index < len(section_lines):
            if section_lines[current_index] == "```":
                example_code_end_found = True
                break
            if "####" in section_lines[current_index]:
                errors.append(
                    f"文件格式错误: {file_path} - 转写示例内容不应包含四级标题"
                )
                break
            current_index += 1

        if not example_code_end_found:
            errors.append(
                f"文件格式错误: {file_path} - 未找到转写示例代码块结束'```'"
            )

        return len(errors) == 0, errors

    def _validate_conditional_rewrite_example(
        self,
        section_lines: list[str],
        current_index: int,
        requires_rewrite_params: set,
        category_dir: str,
        file_path: str,
        section_index: int,
    ) -> tuple[bool, list[str]]:
        """校验条件性转写示例"""
        errors = []

        if (
            current_index < len(section_lines)
            and section_lines[current_index] == "### 转写示例"
        ):
            # 检查四级标题
            found_4th_level = False
            param_headers = set()
            temp_index = current_index + 1

            while temp_index < len(section_lines) and not section_lines[
                temp_index
            ].startswith("### "):
                if section_lines[temp_index].startswith("#### "):
                    found_4th_level = True
                    header_content = section_lines[temp_index][
                        5:
                    ].strip()  # 去除"#### "
                    param_headers.add(header_content)
                temp_index += 1

            # 检查是否需要四级标题
            missing_params = []
            if requires_rewrite_params:
                if not found_4th_level:
                    errors.append(
                        f"文件格式错误: {file_path} - 转写示例应包含参数四级标题"
                    )
                else:
                    # 检查是否包含了所有需要转写的参数
                    for param in requires_rewrite_params:
                        missing = True
                        for header in param_headers:
                            if param in header:
                                missing = False
                                break
                        if missing:
                            missing_params.append(param)
                    if len(missing_params) > 0:
                        errors.append(
                            f"文件格式错误: {file_path} - 缺少参数四级标题: {missing_params}"
                        )
            elif found_4th_level and category_dir not in [
                "paddle_more_args",
                "args_default_value_diff",
            ]:
                errors.append(
                    f"文件格式错误: {file_path} - 不应包含四级标题但存在"
                )

        elif requires_rewrite_params:
            errors.append(f"文件格式错误: {file_path} - 需要转写示例但未找到")

        return len(errors) == 0, errors

    def validate_section(
        self,
        section_lines: list[str],
        category_dir: str,
        torch_api_name: str,
        file_path: str,
        section_index: int,
    ) -> tuple[bool, list[str]]:
        """校验单个部分（普通部分或重载部分）"""
        errors = []
        current_index = 0
        torch_params = None
        paddle_params = None

        try:
            # 1. 检查torch超链接并提取API名称
            if not self._validate_torch_header(
                section_lines,
                current_index,
                torch_api_name,
                file_path,
                section_index,
                errors,
            ):
                return False, errors

            # 提取 torch API 名称
            current_index += 1

            # 2. 检查torch代码块并提取参数
            if not self._validate_code_block(
                section_lines,
                current_index,
                "python",
                file_path,
                section_index,
                errors,
                "torch",
            ):
                return False, errors

            # 从 torch 代码块中提取参数列表（可能是多行）
            code_start = current_index + 1
            code_end = self._find_code_block_end(section_lines, current_index)
            if code_start < code_end:
                # 提取代码块中的所有行，合并成一个字符串
                torch_signature_lines = []
                for idx in range(code_start, code_end):
                    if idx < len(section_lines):
                        line = section_lines[idx].strip()
                        if line:
                            torch_signature_lines.append(line)
                if torch_signature_lines:
                    # 将多行合并，去除换行和多余空格
                    torch_signature = " ".join(torch_signature_lines)
                    torch_params = self.parse_parameters_from_signature(
                        torch_signature
                    )

            current_index = code_end + 1

            # 3. 检查paddle部分并提取API名称和参数
            composite_implement, have_paddle_section, current_index, success = (
                self._validate_paddle_section(
                    section_lines,
                    current_index,
                    category_dir,
                    file_path,
                    section_index,
                    errors,
                )
            )

            # 如果有 paddle 部分，提取 paddle API 名称和参数
            if have_paddle_section:
                # 回溯找到 paddle API 行和代码块
                paddle_api_line_idx = -1
                for i in range(current_index - 1, -1, -1):
                    if section_lines[i].startswith("### [") and i > 0:
                        paddle_api_line_idx = i
                        break

                # 从 paddle 代码块中提取参数
                if paddle_api_line_idx >= 0:
                    # paddle 代码块应该在 API 行之后
                    paddle_code_start = (
                        paddle_api_line_idx + 2
                    )  # 跳过 API 行和 ```python
                    if paddle_code_start < len(section_lines):
                        # 可能有多行，需要合并
                        paddle_signature_lines = []
                        idx = paddle_code_start
                        while (
                            idx < len(section_lines)
                            and section_lines[idx] != "```"
                        ):
                            line = section_lines[idx].strip()
                            if line:
                                paddle_signature_lines.append(line)
                            idx += 1
                        if paddle_signature_lines:
                            paddle_signature = " ".join(paddle_signature_lines)
                            paddle_params = (
                                self.parse_parameters_from_signature(
                                    paddle_signature
                                )
                            )

            if not success:
                return False, errors

            # 4. 检查总结行
            current_index = self._skip_to_next_header(
                section_lines, current_index
            )

            # 5. 检查参数映射表格
            requires_rewrite_params = set()
            if (
                category_dir not in ["invok_only_diff", "composite_implement"]
            ) or (composite_implement and have_paddle_section):
                if not self._validate_parameter_table_header(
                    section_lines,
                    current_index,
                    file_path,
                    section_index,
                    errors,
                ):
                    return False, errors
                current_index += 1

                # 提取表格内容并校验
                table_lines = []
                while (
                    current_index < len(section_lines)
                    and "|" in section_lines[current_index]
                ):
                    table_lines.append(section_lines[current_index])
                    current_index += 1

                rewrite_params, table_errors = (
                    self.validate_parameter_table_remarks(
                        table_lines,
                        category_dir,
                        file_path,
                        section_index,
                        torch_params,
                        paddle_params,
                    )
                )
                requires_rewrite_params = rewrite_params
                errors.extend(table_errors)

            # 6. 检查转写示例
            rewrite_success, rewrite_errors = self.validate_rewrite_example(
                section_lines,
                current_index,
                requires_rewrite_params,
                category_dir,
                file_path,
                section_index,
            )
            errors.extend(rewrite_errors)

            return len(errors) == 0, errors

        except Exception as e:
            error_info = traceback.format_exc()
            current_line = (
                section_lines[current_index]
                if current_index < len(section_lines)
                else "EOF"
            )
            errors.append(
                f"校验过程中发生异常: {file_path} - 异常类型: {type(e).__name__}, 异常信息: {e!s}"
            )
            return False, errors

    def _validate_torch_header(
        self,
        section_lines: list[str],
        current_index: int,
        torch_api_name: str,
        file_path: str,
        section_index: int,
        errors: list[str],
    ) -> bool:
        """校验torch头部分"""
        if current_index >= len(section_lines) or not re.match(
            r"^### \[" + re.escape(torch_api_name) + r"\]\(.*\)$",
            section_lines[current_index],
        ):
            current_line = (
                section_lines[current_index]
                if current_index < len(section_lines)
                else "EOF"
            )
            errors.append(
                f"文件格式错误: {file_path} - 应为torch超链接: '{current_line}'"
            )
            return False

        self._validate_api_url(
            section_lines[current_index],
            torch_api_name,
            file_path,
            errors,
        )
        return True

    def _validate_code_block(
        self,
        section_lines: list[str],
        current_index: int,
        language: str,
        file_path: str,
        section_index: int,
        errors: list[str],
        block_type: str,
    ) -> bool:
        """校验代码块"""
        expected_start = f"```{language}"
        if (
            current_index >= len(section_lines)
            or section_lines[current_index] != expected_start
        ):
            current_line = (
                section_lines[current_index]
                if current_index < len(section_lines)
                else "EOF"
            )
            errors.append(
                f"文件格式错误: {file_path} - 应为'{expected_start}'，当前行: '{current_line}'"
            )
            return False
        return True

    def _find_code_block_end(
        self, section_lines: list[str], start_index: int
    ) -> int:
        """查找代码块结束位置"""
        current_index = start_index + 1
        while current_index < len(section_lines):
            if section_lines[current_index] == "```":
                return current_index
            current_index += 1
        return len(section_lines)

    def _validate_paddle_section(
        self,
        section_lines: list[str],
        current_index: int,
        category_dir: str,
        file_path: str,
        section_index: int,
        errors: list[str],
    ) -> tuple[bool, bool, int, bool]:
        """校验paddle部分"""
        composite_implement = category_dir == "composite_implement"
        have_paddle_section = False

        if (
            composite_implement
            and current_index < len(section_lines)
            and not re.match(
                r"^### \[.*\]\(.*\)$", section_lines[current_index]
            )
        ):
            return composite_implement, False, current_index, True

        if category_dir == "invok_only_diff":
            # invok_only_diff 可选paddle部分
            if current_index < len(section_lines) and re.match(
                r"^### \[.*\]\(.*\)$", section_lines[current_index]
            ):
                return self._process_paddle_section(
                    section_lines,
                    current_index,
                    file_path,
                    section_index,
                    errors,
                    composite_implement,
                )
            else:
                return composite_implement, False, current_index, True
        else:
            # 其他类别必须包含paddle部分
            return self._process_required_paddle_section(
                section_lines,
                current_index,
                file_path,
                section_index,
                errors,
                composite_implement,
            )

    def _process_paddle_section(
        self,
        section_lines: list[str],
        current_index: int,
        file_path: str,
        section_index: int,
        errors: list[str],
        composite_implement: bool,
    ) -> tuple[bool, bool, int, bool]:
        """处理可选的paddle部分"""
        paddle_line = section_lines[current_index]
        paddle_api_match = re.search(r"\[(.*?)\]", paddle_line)
        if paddle_api_match:
            paddle_api_name = paddle_api_match.group(1)
            self._validate_api_url(
                paddle_line, paddle_api_name, file_path, errors
            )

        current_index += 1
        if not self._validate_code_block(
            section_lines,
            current_index,
            "python",
            file_path,
            section_index,
            errors,
            "paddle",
        ):
            return composite_implement, True, current_index, False
        current_index += 1

        current_index = (
            self._find_code_block_end(section_lines, current_index) + 1
        )
        return composite_implement, True, current_index, True

    def _process_required_paddle_section(
        self,
        section_lines: list[str],
        current_index: int,
        file_path: str,
        section_index: int,
        errors: list[str],
        composite_implement: bool,
    ) -> tuple[bool, bool, int, bool]:
        """处理必须的paddle部分"""
        if current_index >= len(section_lines) or not re.match(
            r"^### \[.*\]\(.*\)$", section_lines[current_index]
        ):
            current_line = (
                section_lines[current_index]
                if current_index < len(section_lines)
                else "EOF"
            )
            errors.append(
                f"文件格式错误: {file_path} - 应为paddle超链接，当前行: '{current_line}'"
            )
            return composite_implement, False, current_index, False

        paddle_line = section_lines[current_index]
        paddle_api_match = re.search(r"\[(.*?)\]", paddle_line)
        if paddle_api_match:
            paddle_api_name = paddle_api_match.group(1)
            self._validate_api_url(
                paddle_line, paddle_api_name, file_path, errors
            )

        current_index += 1
        if not self._validate_code_block(
            section_lines,
            current_index,
            "python",
            file_path,
            section_index,
            errors,
            "paddle",
        ):
            return composite_implement, True, current_index, False
        current_index += 1

        current_index = (
            self._find_code_block_end(section_lines, current_index) + 1
        )
        return composite_implement, True, current_index, True

    def _skip_to_next_header(
        self, section_lines: list[str], current_index: int
    ) -> int:
        """跳转到下一个标题"""
        while current_index < len(section_lines) and not section_lines[
            current_index
        ].startswith("#"):
            current_index += 1
        return current_index

    def _validate_parameter_table_header(
        self,
        section_lines: list[str],
        current_index: int,
        file_path: str,
        section_index: int,
        errors: list[str],
    ) -> bool:
        """校验参数映射表头"""
        if (
            current_index >= len(section_lines)
            or section_lines[current_index] != "### 参数映射"
        ):
            current_line = (
                section_lines[current_index]
                if current_index < len(section_lines)
                else "EOF"
            )
            errors.append(
                f"文件格式错误: {file_path} - 缺少'### 参数映射'标题，当前行: '{current_line}'"
            )
            return False
        return True

    def _validate_api_url(
        self,
        line: str,
        api_name: str,
        file_path: str,
        errors: list[str],
    ) -> None:
        """验证API URL是否正确"""
        # 提取当前URL
        match = re.search(r"\]\((.*?)\)", line)
        if not match:
            return

        current_url = match.group(1)
        # 获取标准URL
        standard_url = get_url(api_name, disable_warning=True)

        if not standard_url:
            # 如果获取不到标准URL，跳过验证
            return

        if current_url != standard_url:
            error_msg = (
                f"URL 链接错误: {file_path}\n"
                f"  API: {api_name}\n"
                f"  当前 URL: {current_url}\n"
                f"  预期 URL: {standard_url}"
            )
            self.url_errors.append(
                {
                    "file_path": file_path,
                    "api_name": api_name,
                    "line": line,
                    "current_url": current_url,
                    "standard_url": standard_url,
                }
            )
            errors.append(error_msg)

    def _fix_url_in_file(self, file_path: str, url_fixes: list[dict]) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            for fix in url_fixes:
                api_name = fix["api_name"]
                current_url = fix["current_url"]
                standard_url = fix["standard_url"]

                escaped_name = re.escape(api_name)
                pattern_name = escaped_name.replace(r"\_", "_").replace(
                    r"_", r"(?:_|\\_)"
                )

                escaped_current_url = re.escape(current_url)

                pattern = f"\\[{pattern_name}\\]\\({escaped_current_url}\\)"

                escaped_output_name = api_name.replace("_", r"\_")
                replacement = f"[{escaped_output_name}]({standard_url})"
                content = re.sub(pattern, replacement, content, count=1)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return True
            return False
        except Exception as e:
            print(f"修复文件 {file_path} 时出错: {e}")
            import traceback

            traceback.print_exc()
            return False

    def process_file(self, file_path: str, category_dir: str) -> bool:
        """处理单个文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip() != ""]
                lines = [line.replace(r"\_", "_") for line in lines]
        except Exception as e:
            error_msg = f"读取文件失败: {file_path} - {e!s}"
            self.errors.append(error_msg)
            return False

        torch_api_name = os.path.basename(file_path)[:-3]  # 去掉.md后缀

        if torch_api_name.startswith("transformers"):
            return True

        # 检查是否为重载API文档
        is_overloaded = len(lines) > 1 and "重载" in lines[1]

        if is_overloaded:
            return self._process_overloaded_file(
                lines, category_dir, torch_api_name, file_path
            )
        else:
            return self._process_regular_file(
                lines, category_dir, torch_api_name, file_path
            )

    def _process_overloaded_file(
        self,
        lines: list[str],
        category_dir: str,
        torch_api_name: str,
        file_path: str,
    ) -> bool:
        """处理重载文件"""
        separator = "-------------------------------------------------------------------------------------------------"
        sections = []
        current_section = []
        in_section = False

        for line in lines:
            if line == separator:
                if current_section:
                    sections.append(current_section)
                    current_section = []
                in_section = True
            elif in_section:
                current_section.append(line)

        if current_section:
            sections.append(current_section)

        if not sections:
            error_msg = (
                f"文件格式错误: {file_path} - 重载文档格式错误，未找到有效部分"
            )
            self.errors.append(error_msg)
            return False

        section_valid = False
        all_errors = []

        for i, section in enumerate(sections):
            if i == 0:
                section_start = 0
                for j, line in enumerate(section):
                    if line.startswith("### ["):
                        section_start = j
                        break
                actual_section = section[section_start:]
            else:
                actual_section = section

            if actual_section:
                valid, section_errors = self.validate_section(
                    actual_section,
                    category_dir,
                    torch_api_name,
                    file_path,
                    i + 1,
                )
                all_errors.extend(section_errors)
                if valid:
                    section_valid = True
                    break

        if not section_valid:
            self.errors.extend(all_errors)
        return section_valid

    def _process_regular_file(
        self,
        lines: list[str],
        category_dir: str,
        torch_api_name: str,
        file_path: str,
    ) -> bool:
        """处理普通文件"""
        current_index = 0
        expected_title = (
            f"## [ {self.CATEGORY_MAP[category_dir]} ]{torch_api_name}"
        )

        if (
            current_index >= len(lines)
            or lines[current_index] != expected_title
        ):
            error_msg = (
                f"文件格式错误: {file_path} - 第一行应为 '{expected_title}'"
            )
            self.errors.append(error_msg)
            return False

        current_index += 1
        valid, errors = self.validate_section(
            lines[current_index:], category_dir, torch_api_name, file_path, 1
        )
        self.errors.extend(errors)
        return valid

    def run_validation(self) -> None:
        """运行校验"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        error_file = os.path.join(script_dir, "api_difference_error.txt")
        # 清空错误日志文件
        for file_path in [error_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

        # 遍历所有类别目录
        for category_dir in self.CATEGORY_MAP:
            category_path = os.path.join(self.api_difference_dir, category_dir)
            if not os.path.isdir(category_path):
                continue

            for file_name in os.listdir(category_path):
                if not file_name.endswith(".md"):
                    continue

                self.total_files += 1
                file_path = os.path.join(category_path, file_name)

                try:
                    if self.process_file(file_path, category_dir):
                        self.valid_files += 1
                except Exception as e:
                    error_msg = f"处理文件时发生异常: {file_path}"
                    self.errors.append(error_msg)

    def print_results(self) -> None:
        """打印结果"""
        print(f"total files: {self.total_files}")
        print(f"valid files: {self.valid_files}")
        acc = (
            f"{self.valid_files / self.total_files:.2%}"
            if self.total_files > 0
            else "N/A"
        )
        print(f"accuracy:\n{acc}")

        if self.url_errors:
            print(f"\nURL错误数量: {len(self.url_errors)}")
            if self.fix_url:
                print(f"已修复的文件数量: {self.url_fixed_count}")

    def fix_urls(self) -> int:
        """修复所有URL错误"""
        if not self.url_errors:
            return 0

        file_fixes = {}
        for error in self.url_errors:
            file_path = error["file_path"]
            if file_path not in file_fixes:
                file_fixes[file_path] = []
            file_fixes[file_path].append(error)

        fixed_count = 0
        for file_path, fixes in file_fixes.items():
            if self._fix_url_in_file(file_path, fixes):
                fixed_count += 1
                print(f"已修复: {file_path}")

        self.url_fixed_count = fixed_count
        return fixed_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="验证API差异文档格式")
    parser.add_argument(
        "--fix-url", action="store_true", help="自动修复错误的API URL"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    api_difference_dir = os.path.join(script_dir, "../api_difference")

    validator = APIDifferenceValidator(api_difference_dir, fix_url=args.fix_url)
    validator.run_validation()

    if args.fix_url and validator.url_errors:
        print("\n开始修复URL错误...")
        fixed_count = validator.fix_urls()
        print(f"\n总共修复了 {fixed_count} 个文件的URL错误")

    # 保存错误信息
    error_file = os.path.join(
        script_dir, "validate_api_difference_format_error.txt"
    )
    if validator.url_errors:
        print("\n" + "=" * 80)
        print("❌ 检测到 API 文档链接校验失败！")
        print("部分 API 的超链接与标准文档链接不一致。")
        print("\n🚀 快速修复方案：请在本地运行以下命令自动修正：")
        print(
            "   python docs/guides/model_convert/convert_from_pytorch/tools/validate_api_difference_format.py --fix-url"
        )
        print("\n或者根据上方的错误日志手动修改。")
        print("=" * 80 + "\n")
    if validator.errors:
        with open(error_file, "w", encoding="utf-8") as f:
            for error in validator.errors:
                if isinstance(error, list):
                    for sub_error in error:
                        print(sub_error)
                        f.write(f"{sub_error}\n")
                        break
                else:
                    print(error)
                    f.write(f"{error}\n")

        print(f"error log saved to: {error_file}")
        validator.print_results()
        return 1
    else:
        if os.path.exists(error_file):
            os.remove(error_file)
        validator.print_results()
        return 0


if __name__ == "__main__":
    sys.exit(main())
