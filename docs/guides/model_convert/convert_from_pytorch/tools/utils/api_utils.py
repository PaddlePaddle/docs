from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from logging import getLogger
from pathlib import Path

from .api_url_parser import get_parser

logger = getLogger(__name__)


def get_url(
    api_name: str, package: str | None = None, disable_warning: bool = False
) -> str:
    return ""
    # TODO(zhwesky2010): Test get_url stability in different environment, will cause instability now
    try:
        api_name = api_name.replace(r"\_", "_")
        if package is None:
            package = api_name.split(".")[0]
        url = get_parser(package).get_api_url(api_name) or ""
        if url == "" and not disable_warning:
            logger.warning("Missing api %s in package %s", api_name, package)
    except Exception as e:
        logger.error("get api %s in package %s error: %s", api_name, package, e)
        return ""
    return url


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
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.dirname(current_script_path)
    base_dir = os.path.dirname(tools_dir)
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
        return {}


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


def extract_no_need_convert_list():
    api_mapping_file_path = Path(__file__).parent.parent / "api_mapping.json"
    attribute_mapping_file_path = (
        Path(__file__).parent.parent / "attribute_mapping.json"
    )
    if not api_mapping_file_path.exists():
        raise FileNotFoundError(
            f"api_mapping.json should exist at {api_mapping_file_path.as_posix()} to extract no_need_convert_list"
        )
    if not attribute_mapping_file_path.exists():
        raise FileNotFoundError(
            f"attribute_mapping.json should exist at {attribute_mapping_file_path.as_posix()} to extract no_need_convert_list"
        )
    with open(api_mapping_file_path, "r", encoding="utf-8") as file:
        api_mapping_json = json.load(file)
    with open(attribute_mapping_file_path, "r", encoding="utf-8") as file:
        attribute_mapping_json = json.load(file)

    no_need_list = [
        k
        for k, v in api_mapping_json.items()
        if v.get("Matcher") == "ChangePrefixMatcher"
    ]
    no_need_list += [
        k
        for k, v in attribute_mapping_json.items()
        if v.get("Matcher") == "ChangePrefixMatcher"
    ]
    if len(no_need_list) == 0:
        raise ValueError("no_need_list is empty")
    return no_need_list
