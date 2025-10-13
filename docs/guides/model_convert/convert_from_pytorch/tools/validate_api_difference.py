import os
import re


def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # API对比文档目录路径
    api_difference_dir = os.path.join(script_dir, "../api_difference")

    # 定义类别映射：英文目录名 -> 中文类别名称
    category_map = {
        "invok_diff_only": "仅 API 调用方式不一致",
        "args_name_diff": "仅参数名不一致",
        "paddle_more_args": "paddle 参数更多",
        "args_default_value_diff": "参数默认值不一致",
        "torch_more_args": "torch 参数更多",
        "input_args_usage_diff": "输入参数用法不一致",
        "input_args_type_diff": "输入参数类型不一致",
        "output_args_type_diff": "返回参数类型不一致",
        "composite_implement": "组合替代实现",
    }

    # 定义每个类别的总结语句
    summary_map = {
        "invok_diff_only": "两者功能一致，但调用方式不一致，具体如下：",
        "args_name_diff": "两者功能一致且参数用法一致，仅参数名不一致，具体如下：",
        "paddle_more_args": "其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：",
        "args_default_value_diff": "两者功能一致且参数用法一致，参数默认值不一致，具体如下：",
        "torch_more_args": "PyTorch 相比 Paddle 支持更多其他参数，具体如下：",
        "input_args_usage_diff": "其中 PyTorch 的输入参数与 Paddle 用法不一致，具体如下：",
        "input_args_type_diff": "两者功能一致但参数类型不一致，具体如下：",
        "output_args_type_diff": "两者功能一致但返回参数类型不同，具体如下：",
        "composite_implement": "Paddle 无此 API，需要组合实现。",
    }

    # 初始化统计变量
    total_files = 0
    valid_files = 0
    errors = []

    # 遍历所有类别目录
    for category_dir in category_map:
        category_path = os.path.join(api_difference_dir, category_dir)
        if not os.path.isdir(category_path):
            continue

        # 遍历目录下的所有.md文件
        for file_name in os.listdir(category_path):
            if not file_name.endswith(".md"):
                continue

            total_files += 1
            file_path = os.path.join(category_path, file_name)
            torch_api_name = file_name[:-3]  # 去掉.md后缀

            # 读取文件内容并移除空行
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip() != ""]
                    lines = [line.replace(r"\_", "_") for line in lines]
            except Exception as e:
                errors.append(f"读取文件失败: {file_path} - {e!s}")
                continue

            current_index = 0
            error_found = False

            # 1. 检查类别标题 (## [ 中文类别 ]torch_api_name)
            expected_title = (
                f"## [ {category_map[category_dir]} ]{torch_api_name}"
            )
            if (
                current_index >= len(lines)
                or lines[current_index] != expected_title
            ):
                error_found = True
                errors.append(
                    f"文件格式错误: {file_path} - 第一行应为 '{expected_title}'"
                )
            current_index += 1

            # 2. 检查torch超链接 (### [torch_api_name](url))
            if current_index >= len(lines) or not re.match(
                r"^### \[" + re.escape(torch_api_name) + r"\]\(.*\)$",
                lines[current_index],
            ):
                error_found = True
                errors.append(
                    f"文件格式错误: {file_path} - 第二行应为torch超链接"
                )
            current_index += 1

            # 3. 检查torch代码块开始 (```python)
            if (
                current_index >= len(lines)
                or lines[current_index] != "```python"
            ):
                error_found = True
                errors.append(
                    f"文件格式错误: {file_path} - 第三行应为'```python'"
                )
            current_index += 1

            while lines[current_index] != "```":
                current_index += 1
                if current_index >= len(lines):
                    error_found = True
                    errors.append(
                        f"文件格式错误: {file_path} - 未找到torch代码块结束'```'"
                    )
                    break
            current_index += 1

            # 6. 检查paddle部分 (非组合实现类别)
            if category_dir != "composite_implement":
                # 6.1 paddle超链接
                if current_index >= len(lines) or not re.match(
                    r"^### \[.*\]\(.*\)$", lines[current_index]
                ):
                    error_found = True
                    errors.append(
                        f"文件格式错误: {file_path} - 第六行应为paddle超链接"
                    )
                current_index += 1

                # 6.2 paddle代码块开始
                if (
                    current_index >= len(lines)
                    or lines[current_index] != "```python"
                ):
                    error_found = True
                    errors.append(
                        f"文件格式错误: {file_path} - 第七行应为'```python'"
                    )
                current_index += 1

                while lines[current_index] != "```":
                    current_index += 1
                    if current_index >= len(lines):
                        error_found = True
                        errors.append(
                            f"文件格式错误: {file_path} - 未找到'```'"
                        )
                        break
                current_index += 1

            # 7. 检查总结行
            expected_summary = summary_map[category_dir]
            # if current_index >= len(lines) or lines[current_index] != expected_summary:
            #     error_found = True
            #     errors.append(f"文件格式错误: {file_path} - 总结行应为 '{expected_summary}'")
            while current_index < len(lines) and not lines[
                current_index
            ].startswith("#"):
                current_index += 1

            # 8. 检查参数映射表格 (非2和10类)
            if category_dir not in ["invok_diff_only", "composite_implement"]:
                # 8.1 参数映射标题
                if (
                    current_index >= len(lines)
                    or lines[current_index] != "### 参数映射"
                ):
                    error_found = True
                    errors.append(
                        f"文件格式错误: {file_path} - 缺少'### 参数映射'标题"
                    )
                current_index += 1

                while (
                    current_index < len(lines) and "|" in lines[current_index]
                ):
                    current_index += 1

            # 9. 检查转写示例
            if category_dir in ["invok_diff_only", "composite_implement"]:
                # 9.1 转写示例标题
                if (
                    current_index >= len(lines)
                    or lines[current_index] != "### 转写示例"
                ):
                    error_found = True
                    errors.append(
                        f"文件格式错误: {file_path} - 缺少'### 转写示例'"
                    )
                current_index += 1

                # 9.2 转写示例代码块
                if (
                    current_index >= len(lines)
                    or lines[current_index] != "```python"
                ):
                    error_found = True
                    errors.append(
                        f"文件格式错误: {file_path} - 转写示例代码块开始错误"
                    )
                current_index += 1

                # 9.3 转写内容 (必须包含PyTorch和Paddle示例)
                # if current_index + 1 >= len(lines):
                #     error_found = True
                #     errors.append(f"文件格式错误: {file_path} - 转写示例内容缺失")
                # else:
                #     pytorch_line = lines[current_index]
                #     paddle_line = lines[current_index + 1]
                #     if not ('# PyTorch' in pytorch_line and '# Paddle' in paddle_line):
                #         error_found = True
                #         errors.append(f"文件格式错误: {file_path} - 转写示例内容缺失PyTorch/Paddle标记")

                #     # 检查内容中不能包含四级标题
                #     if '####' in pytorch_line or '####' in paddle_line:
                #         error_found = True
                #         errors.append(f"文件格式错误: {file_path} - 转写示例内容不应包含四级标题")

                while (
                    current_index < len(lines) and lines[current_index] != "```"
                ):
                    current_index += 1
                    if current_index >= len(lines):
                        error_found = True
                        errors.append(
                            f"文件格式错误: {file_path} - 未找到'```'"
                        )
                        break
                    if "####" in lines[current_index]:
                        error_found = True
                        errors.append(
                            f"文件格式错误: {file_path} - 转写示例内容不应包含四级标题"
                        )

                current_index += 1

            elif category_dir in [
                "torch_more_args",
                "input_args_usage_diff",
                "input_args_type_diff",
                "output_args_type_diff",
                "paddle_more_args",
                "args_default_value_diff",
            ]:
                # 9.1 转写示例标题
                if (
                    current_index >= len(lines)
                    or lines[current_index] != "### 转写示例"
                ):
                    if category_dir not in [
                        "paddle_more_args",
                        "args_default_value_diff",
                    ]:
                        error_found = True
                        errors.append(
                            f"文件格式错误: {file_path} - 缺少'### 转写示例'"
                        )
                current_index += 1

                # 9.2 检查至少一个四级标题 (####)
                found_4th_level = False
                for i in range(current_index, len(lines)):
                    if lines[i].startswith("####"):
                        found_4th_level = True
                        break
                if not found_4th_level:
                    if category_dir not in [
                        "paddle_more_args",
                        "args_default_value_diff",
                    ]:
                        error_found = True
                        errors.append(
                            f"文件格式错误: {file_path} - 转写示例应包含至少一个四级标题"
                        )

            else:  # 类别3: args_name_diff
                # 检查不应存在转写示例
                if (
                    current_index < len(lines)
                    and lines[current_index] == "### 转写示例"
                ):
                    error_found = True
                    errors.append(
                        f"文件格式错误: {file_path} - 类别3不应包含'### 转写示例'"
                    )

            # 更新有效文件计数
            if not error_found:
                valid_files += 1

    # 输出统计信息
    print(f"总文件数: {total_files}")
    print(f"有效文件数: {valid_files}")
    print(f"通过率: {valid_files / total_files:.2%}")

    # 保存错误信息到文件
    error_file = os.path.join(script_dir, "api_difference_error.txt")
    with open(error_file, "w", encoding="utf-8") as f:
        f.writelines(error + "\n" for error in errors)

    # 如果没有错误，删除错误文件
    if not errors:
        os.remove(error_file)


if __name__ == "__main__":
    main()
