import argparse
import concurrent.futures
import os
import random
import re
import time
from collections import defaultdict
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm  # 用于显示进度条
from urllib3.util.retry import Retry

# 默认文件路径
DEFAULT_FILE_PATH = "/workspace/paddleDocs/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.md"

USER_AGENT = ""

# 重试策略配置
RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET"],
)


def create_session():
    """创建带有重试机制的会话"""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def parse_toc(lines):
    """
    解析目录表格，返回目录条目列表和结束行号
    """
    toc = []
    toc_end_line = -1
    in_toc = False
    found_header = False

    for i, line in enumerate(lines):
        line = line.strip()

        if line == "## API 映射分类":
            in_toc = True
            continue

        if in_toc:
            if line.startswith("|") and "序号" in line and "类别" in line:
                found_header = True
                continue

            if (
                found_header
                and line.startswith("|")
                and re.match(r"^\|?[-:\s|]+\|?$", line)
            ):
                continue  # 跳过分隔行

            if found_header and line.startswith("|"):
                # 解析数据行
                columns = [col.strip() for col in line.split("|")[1:-1]]
                if len(columns) >= 2:
                    toc.append((columns[0], columns[1]))  # (序号, 类别名称)
            else:
                # 表格结束
                if found_header:
                    toc_end_line = i
                    break

    return toc, toc_end_line


def parse_categories(lines):
    """
    解析类别部分，提取所有类别及其表格数据
    """
    categories = []
    current_category = None
    current_table = []
    in_table = False

    for line in lines:
        line = line.strip()

        # 检测类别标题 (## X. 类别名称)
        category_match = re.match(r"^### (\d+)\. (.+)$", line)
        if category_match:
            # 保存前一个类别和表格
            if current_category is not None:
                categories.append(
                    {
                        "id": current_category["id"],
                        "name": current_category["name"],
                        "table": current_table,
                    }
                )

            # 开始新类别
            current_category = {
                "id": int(category_match.group(1)),
                "name": category_match.group(2),
            }
            current_table = []
            in_table = False
            continue

        # 检测表格开始
        if line.startswith("|") and "Pytorch" in line and "Paddle" in line:
            in_table = True
            continue

        # 检测表格行
        if in_table and line.startswith("|"):
            # 跳过表头分隔行
            if re.match(r"^\|?[-:\s|]+\|?$", line):
                continue

            # 解析表格行
            columns = [col.strip() for col in line.split("|")[1:-1]]
            if len(columns) >= 4:
                current_table.append(
                    {
                        "index": columns[0],
                        "pytorch": columns[1],
                        "paddle": columns[2],
                        "note": columns[3] if len(columns) > 3 else "",
                    }
                )
            continue

        # 检测表格结束
        if in_table and not line.startswith("|") and line != "":
            in_table = False

    # 添加最后一个类别
    if current_category is not None:
        categories.append(
            {
                "id": current_category["id"],
                "name": current_category["name"],
                "table": current_table,
            }
        )

    return categories


def extract_links(text):
    """
    从文本中提取所有超链接
    """
    return re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)


def check_toc_consistency(toc, categories):
    """
    检查目录与类别标题的一致性
    """
    warnings = []

    # 检查数量是否一致
    if len(toc) != len(categories):
        warnings.append(
            f"目录中有 {len(toc)} 个类别，但实际找到 {len(categories)} 个类别"
        )

    # 检查每个类别的序号和名称是否匹配
    for i, (toc_index, toc_name) in enumerate(toc):
        if i >= len(categories):
            break

        cat = categories[i]
        if str(cat["id"]) != toc_index:
            warnings.append(
                f"目录中第 {i + 1} 个类别序号为 {toc_index}，但实际类别序号为 {cat['id']}"
            )

        if cat["name"] != toc_name:
            warnings.append(
                f"目录中第 {i + 1} 个类别名称为 '{toc_name}'，但实际类别名称为 '{cat['name']}'"
            )

    return warnings


def check_unique_torch_apis(categories):
    """
    检查 torch API 是否唯一
    """
    torch_apis = defaultdict(list)  # api -> [category_ids]

    for category in categories:
        for row in category["table"]:
            links = extract_links(row["pytorch"])
            if links:
                api_name = links[0][0]  # 取第一个链接的文本作为 API 名称
                torch_apis[api_name].append(category["id"])

    # 检查重复的 API
    warnings = []
    for api, category_ids in torch_apis.items():
        if len(category_ids) > 1:
            warning_msg = f"Torch API '{api}' 在多个类别中出现: {category_ids}"
            warnings.append(warning_msg)

    return warnings


def check_links_exist(categories):
    """
    检查必要的超链接是否存在（根据新规则）
    规则：
    1. 第二列(Pytorch)必须有超链接
    2. 第三列(Paddle):
       - 对于"组合替代实现"、"可删除"、"功能缺失"类别，如果内容为空或"-"则不检查
       - 否则必须有超链接
    3. 第四列(备注): 除了"API完全一致类别"（类别1）外，都需要有超链接
    """
    warnings = []

    for category in categories:
        category_id = category["id"]
        category_name = category["name"]

        for i, row in enumerate(category["table"]):
            row_num = i + 1  # 行号从1开始

            # 1. 检查第二列 (Pytorch) 必须有超链接
            pytorch_links = extract_links(row["pytorch"])
            if not pytorch_links:
                warning_msg = f"类别 {category_id}({category_name}) 第 {row_num} 行第二列缺少超链接: {row['pytorch']}"
                warnings.append(warning_msg)

            # 2. 检查第三列 (Paddle)
            paddle_content = row["paddle"].strip()
            paddle_links = extract_links(row["paddle"])

            # 特殊处理：组合替代实现、可删除、功能缺失
            special_cases = ["组合替代实现", "可删除", "功能缺失"]
            is_special_case = any(
                case in category_name for case in special_cases
            )

            if is_special_case:
                # 对于特殊类别，只有内容不为空且不是"-"时才检查链接
                if (
                    paddle_content
                    and paddle_content != "-"
                    and not paddle_links
                ):
                    warning_msg = f"类别 {category_id}({category_name}) 第 {row_num} 行第三列内容不为空但缺少超链接: {row['paddle']}"
                    warnings.append(warning_msg)
            else:
                # 对于其他类别，必须有超链接
                if not paddle_links:
                    warning_msg = f"类别 {category_id}({category_name}) 第 {row_num} 行第三列缺少超链接: {row['paddle']}"
                    warnings.append(warning_msg)

            # 3. 检查第四列 (备注)
            note_links = extract_links(row["note"])
            note_content = row["note"].strip()

            # 除了类别1（API完全一致类别）外，都需要有超链接
            if (
                category_id != 1
                and note_content
                and note_content != "-"
                and not note_links
            ):
                warning_msg = f"类别 {category_id}({category_name}) 第 {row_num} 行第四列缺少超链接: {row['note']}"
                warnings.append(warning_msg)

    return warnings


def extract_all_urls(categories):
    """
    从所有类别中提取所有URL及其上下文信息
    """
    urls_with_context = []

    for category in categories:
        for row_idx, row in enumerate(category["table"]):
            # 提取Pytorch列链接
            pytorch_links = extract_links(row["pytorch"])
            for link_text, url in pytorch_links:
                urls_with_context.append(
                    {
                        "url": url,
                        "category_id": category["id"],
                        "category_name": category["name"],
                        "row_index": row_idx + 1,
                        "column": "Pytorch",
                        "context": f"{link_text} (类别 {category['id']}.{category['name']} 第 {row_idx + 1} 行)",
                    }
                )

            # 提取Paddle列链接
            paddle_links = extract_links(row["paddle"])
            for link_text, url in paddle_links:
                urls_with_context.append(
                    {
                        "url": url,
                        "category_id": category["id"],
                        "category_name": category["name"],
                        "row_index": row_idx + 1,
                        "column": "Paddle",
                        "context": f"{link_text} (类别 {category['id']}.{category['name']} 第 {row_idx + 1} 行)",
                    }
                )

            # 提取Note列链接
            note_links = extract_links(row["note"])
            for link_text, url in note_links:
                urls_with_context.append(
                    {
                        "url": url,
                        "category_id": category["id"],
                        "category_name": category["name"],
                        "row_index": row_idx + 1,
                        "column": "Note",
                        "context": f"{link_text} (类别 {category['id']}.{category['name']} 第 {row_idx + 1} 行)",
                    }
                )

    return urls_with_context


def is_valid_url(url):
    """
    检查URL是否有效
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def check_url_exists(url_info, session=None):
    """
    检查URL是否存在（是否返回404）
    返回状态码和错误信息
    """
    url = url_info["url"]

    # 检查URL是否有效
    if not is_valid_url(url):
        return {
            "status": "invalid",
            "status_code": None,
            "error": "无效的URL格式",
            "url_info": url_info,
        }

    # 添加随机延迟，避免请求过于频繁
    time.sleep(random.uniform(0.5, 1.5))

    # 创建会话（如果未提供）
    if session is None:
        session = create_session()

    try:
        # 发送HEAD请求（更快，节省带宽）
        response = session.head(url, timeout=10, allow_redirects=True)
        status_code = response.status_code

        # 如果HEAD请求不被支持（405错误），则尝试GET请求
        if status_code == 405:
            response = session.get(url, timeout=10, allow_redirects=True)
            status_code = response.status_code

        # 根据状态码判断URL是否存在
        if status_code == 200:
            return {
                "status": "ok",
                "status_code": status_code,
                "error": None,
                "url_info": url_info,
            }
        elif status_code == 404:
            return {
                "status": "not_found",
                "status_code": status_code,
                "error": "页面不存在",
                "url_info": url_info,
            }
        elif 400 <= status_code < 500:
            return {
                "status": "client_error",
                "status_code": status_code,
                "error": "客户端错误",
                "url_info": url_info,
            }
        elif 500 <= status_code < 600:
            return {
                "status": "server_error",
                "status_code": status_code,
                "error": "服务器错误",
                "url_info": url_info,
            }
        else:
            return {
                "status": "other_error",
                "status_code": status_code,
                "error": f"HTTP状态码: {status_code}",
                "url_info": url_info,
            }

    except requests.exceptions.RequestException as e:
        return {
            "status": "request_error",
            "status_code": None,
            "error": str(e),
            "url_info": url_info,
        }


def check_urls_exist(urls_with_context, max_workers=10):
    """
    使用多线程检查所有URL是否存在（是否返回404）
    返回警告列表
    """
    warnings = []

    urls_with_context = urls_with_context[-700:]

    total_urls = len(urls_with_context)

    print(
        f"开始使用多线程检查 {total_urls} 个URL的存在性（线程数：{max_workers}）..."
    )

    with (
        tqdm(total=total_urls, desc="检查URL") as pbar,
        concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor,
    ):
        # 为每个线程创建一个会话
        sessions = [create_session() for _ in range(max_workers)]

        # 提交所有任务
        future_to_url = {}
        for i, url_info in enumerate(urls_with_context):
            # 分配会话给任务（轮询方式）
            session = sessions[i % max_workers]
            future = executor.submit(check_url_exists, url_info, session)
            future_to_url[future] = url_info

        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()

            # 更新进度条
            pbar.update(1)

            # 如果不是200状态码，则添加到警告列表
            if result["status"] != "ok":
                warning_msg = (
                    f"URL访问错误: {result['error']}\n"
                    f"URL: {result['url_info']['url']}\n"
                    f"上下文: {result['url_info']['context']}\n"
                )
                if result["status_code"]:
                    warning_msg += f"状态码: {result['status_code']}\n"
                warnings.append(warning_msg)

    # 关闭所有会话
    for session in sessions:
        session.close()

    print(f"URL检查完成，发现 {len(warnings)} 个问题")
    return warnings


def main():
    parser = argparse.ArgumentParser(description="Markdown 文件校验工具")
    parser.add_argument(
        "--file",
        "-f",
        help="要校验的 Markdown 文件路径",
        default=DEFAULT_FILE_PATH,
    )
    parser.add_argument(
        "--skip-url-check",
        action="store_true",
        help="跳过URL存在性检查（避免网络请求）",
    )

    args = parser.parse_args()

    current_script_path = os.path.abspath(__file__)
    tools_dir = os.path.dirname(current_script_path)
    base_dir = os.path.dirname(tools_dir)  # 上一级目录

    # 确定要校验的文件路径
    if args.file:
        md_file_path = os.path.abspath(args.file)
    else:
        # 默认文件路径：上一级目录中的 pytorch_api_mapping_cn.md
        md_file_path = os.path.join(base_dir, "pytorch_api_mapping_cn.md")

    # 检查文件是否存在
    if not os.path.exists(md_file_path):
        print(f"错误: 文件 '{md_file_path}' 不存在")
        print("请使用 --file 参数指定文件路径")
        return

    # 读取文件所有行
    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"错误: 读取文件 '{md_file_path}' 时出错: {e!s}")
        return

    # 解析目录表格
    toc, toc_end_line = parse_toc(lines)

    # 解析类别部分（从目录结束行之后开始）
    if toc_end_line != -1:
        category_lines = lines[toc_end_line + 1 :]
    else:
        category_lines = lines
    categories = parse_categories(category_lines)

    print(f"正在校验文件: {md_file_path}")
    print(f"找到 {len(toc)} 个目录条目")
    print(f"找到 {len(categories)} 个类别")

    # 执行三个基本校验
    toc_warnings = check_toc_consistency(toc, categories)
    unique_warnings = check_unique_torch_apis(categories)
    link_warnings = check_links_exist(categories)

    # 输出警告到文件
    if toc_warnings:
        output_path = os.path.join(tools_dir, "toc_warnings.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("目录一致性校验警告:\n")
            f.writelines(warning + "\n" for warning in toc_warnings)
        print(f"生成 {output_path}，包含 {len(toc_warnings)} 个警告")

    if unique_warnings:
        output_path = os.path.join(tools_dir, "unique_warnings.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Torch API 唯一性校验警告:\n")
            f.writelines(warning + "\n" for warning in unique_warnings)
        print(f"生成 {output_path}，包含 {len(unique_warnings)} 个警告")

    if link_warnings:
        output_path = os.path.join(tools_dir, "link_warnings.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("超链接存在性校验警告:\n")
            f.writelines(warning + "\n" for warning in link_warnings)
        print(f"生成 {output_path}，包含 {len(link_warnings)} 个警告")

    # 执行URL存在性检查（除非明确跳过）
    url_warnings = []
    if not args.skip_url_check:
        # 提取所有URL
        urls_with_context = extract_all_urls(categories)
        print(f"找到 {len(urls_with_context)} 个URL需要检查")

        # 检查URL存在性（使用多线程）
        url_warnings = check_urls_exist(
            urls_with_context, max(os.cpu_count() - 4, 1)
        )

        if url_warnings:
            output_path = os.path.join(tools_dir, "url_warnings.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("URL存在性校验警告:\n")
                f.writelines(warning + "\n" for warning in url_warnings)
            print(f"生成 {output_path}，包含 {len(url_warnings)} 个警告")
    else:
        print("跳过URL存在性检查")

    # 如果没有警告，输出成功信息
    if (
        not toc_warnings
        and not unique_warnings
        and not link_warnings
        and not url_warnings
    ):
        print("所有校验通过，没有发现警告!")


if __name__ == "__main__":
    main()
