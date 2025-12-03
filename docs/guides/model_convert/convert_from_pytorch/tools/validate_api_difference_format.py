from __future__ import annotations

import os
import re
import sys
import traceback


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

    def __init__(self, api_difference_dir: str):
        self.api_difference_dir = api_difference_dir
        self.total_files = 0
        self.valid_files = 0
        self.errors = []

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
    ) -> tuple[bool, list[str]]:
        """校验参数映射表格的备注列"""
        errors = []
        requires_rewrite_params = set()

        # 跳过表头行（前两行）
        for i, line in enumerate(table_lines[2:], start=2):
            if not line.strip() or not line.startswith("|"):
                continue

            # 解析表格行
            parts = [part.strip() for part in line.split("|") if part.strip()]
            if len(parts) < 3:
                continue

            pytorch_param, paddle_param, remark = parts[0], parts[1], parts[2]

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
                # else:
                #     requires_rewrite_params.add(paddle_param)

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

        try:
            # 1. 检查torch超链接
            if not self._validate_torch_header(
                section_lines,
                current_index,
                torch_api_name,
                file_path,
                section_index,
                errors,
            ):
                return False, errors
            current_index += 1

            # 2. 检查torch代码块
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
            current_index = (
                self._find_code_block_end(section_lines, current_index) + 1
            )

            # 3. 检查paddle部分
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
                        table_lines, category_dir, file_path, section_index
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


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    api_difference_dir = os.path.join(script_dir, "../api_difference")

    validator = APIDifferenceValidator(api_difference_dir)
    validator.run_validation()

    # 保存错误信息
    error_file = os.path.join(
        script_dir, "validate_api_difference_format_error.txt"
    )
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
