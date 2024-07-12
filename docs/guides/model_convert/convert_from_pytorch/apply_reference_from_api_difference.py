import os
import re
import sys

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)
print(script_dir)

from validate_mapping_in_api_difference import (
    DiffMeta,
    get_meta_from_diff_file,
    process_mapping_index as reference_mapping_item,
)


def mapping_type_to_description(mapping_type):
    mapping_type_1 = [
        "无参数",
        "参数完全一致",
        "仅参数名不一致",
        "仅 paddle 参数更多",
        "仅参数默认值不一致",
    ]

    if mapping_type in mapping_type_1:
        return "功能一致，" + mapping_type, True

    mapping_type_2 = ["torch 参数更多"]
    if mapping_type in mapping_type_2:
        return "功能一致，" + mapping_type, True

    mapping_type_3 = [
        # "参数不一致",
        "返回参数类型不一致",
        "输入参数类型不一致",
        "输入参数用法不一致",
    ]
    if mapping_type in mapping_type_3:
        return "功能一致，" + mapping_type, True

    mapping_type_4 = ["组合替代实现"]
    if mapping_type in mapping_type_4:
        return "组合替代实现", True

    mapping_type_5 = ["涉及上下文修改"]
    if mapping_type in mapping_type_5:
        return "功能一致，" + mapping_type, True

    mapping_type_6 = ["对应 API 不在主框架"]
    if mapping_type in mapping_type_6:
        return "对应 API 不在主框架【占位】", False

    mapping_type_7 = ["功能缺失"]
    if mapping_type in mapping_type_7:
        return "功能缺失", False

    mapping_type_delete = ["可删除"]
    if mapping_type in mapping_type_delete:
        return "无对应 API，可以直接删除，对网络一般无影响", False

    raise ValueError(
        f"Unexpected pyTorch-PaddlePaddle api mapping type {mapping_type}, please check  "
    )
    return "【未知类型】", False


# 以后没有 REFERENCE-ITEM 需要维护了，全部从 api_difference/ 目录生成
_REFERENCE_ITEM_PATTERN = re.compile(
    r"^\| *REFERENCE-MAPPING-ITEM\( *(?P<torch_api>[^,]+) *, *(?P<diff_url>.+) *\) *\|$"
)
REFERENCE_TABLE_PATTERN = re.compile(
    r"^\| *REFERENCE-MAPPING-TABLE\( *(?P<api_prefix>[^,]+) *(, *max_depth *= *(?P<max_depth>\d+) *)?\) *\|$"
)
ALIAS_PATTERN = re.compile(
    r"^\| *ALIAS-REFERENCE-ITEM\( *(?P<alias_name>[^,]+) *, *(?P<torch_api>[^,]+) *\) *\|$"
)
NOT_IMPLEMENTED_PATTERN = re.compile(
    r"^\| *NOT-IMPLEMENTED-ITEM\( *(?P<torch_api>[^,]+) *, *(?P<torch_api_url>.+) *\) *\|$"
)
MANUAL_MAINTAINING_PATTERN = re.compile(
    r"^\| *MANUAL_MAINTAINING-ITEM\(*(?P<torch_api>[^,]+) *,*(?P<torch_url>[^,]+) *, *(?P<paddle_api>[^,]+) *,*(?P<paddle_url>[^,]+) *, *(?P<mapping_type_desc>[^,]+) *, *(?P<diff_url>.+) *\) *\|$"
)


DOCS_REPO_BASEURL = "https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/"


def docs_url_to_relative_page(url):
    """将映射文档的 PaddlePaddle/docs url 转换为网页路径"""
    if not url.startswith(DOCS_REPO_BASEURL):
        return url

    md_path = url[len(DOCS_REPO_BASEURL) :]
    if md_path.endswith(".md"):
        return md_path[:-3] + ".html"
    return md_path


def doc_path_to_relative_page(path):
    """将映射文档的本地路径转换为网页相对路径"""
    md_path = os.path.relpath(path, script_dir)

    assert md_path.endswith(".md"), f"Unexpected mapping doc path: {path}"

    return md_path[:-3] + ".html"


def reference_table_match_to_condition(m):
    api_prefix = m["api_prefix"].strip("`")
    max_depth = m["max_depth"]
    if max_depth is None:
        max_depth = 255
    else:
        max_depth = int(max_depth)
    return api_prefix, max_depth


def get_referenced_api_columns(torch_api, metadata_dict, alias=None):
    assert (
        torch_api in metadata_dict
    ), f'Error: cannot find mapping doc of api "{torch_api}"'
    api_data: DiffMeta = metadata_dict[torch_api]

    diff_page_url = doc_path_to_relative_page(api_data["source_file"])

    torch_api_url = api_data["torch_api_url"]
    api_disp_name = torch_api if alias is None else alias
    torch_api_column = f"[`{api_disp_name}`]({torch_api_url})"

    mapping_type = api_data["mapping_type"]
    mapping_type_column = mapping_type

    _mapping_type_desc, show_diff_url = mapping_type_to_description(
        mapping_type
    )
    desc_column = ""
    if show_diff_url:
        desc_column = f"[详细对比]({diff_page_url})"
        if alias is not None:
            desc_column = f"`{torch_api}` 别名，{desc_column}"

    if "paddle_api" not in api_data:
        if mapping_type not in ["组合替代实现", "可删除", "功能缺失"]:
            print(f"Error: cannot find paddle_api for torch_api: {torch_api}")
        paddle_api_column = ""
    else:
        paddle_api = api_data["paddle_api"]
        paddle_api_url = api_data["paddle_api_url"]
        paddle_api_column = f"[`{paddle_api}`]({paddle_api_url})"

    return [
        torch_api_column,
        paddle_api_column,
        mapping_type_column,
        desc_column,
    ]


def apply_reference_to_row_ex(line, metadata_dict, context, line_idx):
    reference_table_match = REFERENCE_TABLE_PATTERN.match(line)
    alias_match = ALIAS_PATTERN.match(line)
    not_implemented_match = NOT_IMPLEMENTED_PATTERN.match(line)
    manual_maintaining_match = MANUAL_MAINTAINING_PATTERN.match(line)

    row_idx_s = str(context["table_row_idx"])

    if reference_table_match:
        condition = reference_table_match_to_condition(reference_table_match)
        api_list = context["c2a_dict"][
            condition
        ]  # 这个键一定存在，否则说明前面出错了
        output_lines = []
        cur_row_idx = context["table_row_idx"]
        for api in api_list:
            content = get_referenced_api_columns(api, metadata_dict)
            content.insert(0, str(cur_row_idx))
            output = "| " + " | ".join(content) + " |\n"
            output_lines.append(output)
            cur_row_idx += 1
        # 因为外面会给 table_row_idx 自动加 1，所以这里减去 1
        context["table_row_idx"] = cur_row_idx - 1
        return output_lines
    elif alias_match:
        alias_name = alias_match["alias_name"].strip("`").replace(r"\_", "_")
        torch_api = alias_match["torch_api"].strip("`").replace(r"\_", "_")

        content = get_referenced_api_columns(
            torch_api, metadata_dict, alias=alias_name
        )

        content.insert(0, row_idx_s)

        output = "| " + " | ".join(content) + " |\n"
        return [output]
    elif not_implemented_match:
        torch_api = (
            not_implemented_match["torch_api"].strip("`").replace(r"\_", "_")
        )
        torch_api_url = not_implemented_match["torch_api_url"].strip()

        torch_api_column = f"[`{torch_api}`]({torch_api_url})"

        paddle_api_column = ""
        mapping_column = "功能缺失"
        mapping_url_column = ""

        content = [
            row_idx_s,
            torch_api_column,
            paddle_api_column,
            mapping_column,
            mapping_url_column,
        ]
        output = "| " + " | ".join(content) + " |\n"
        return [output]

    elif manual_maintaining_match:
        torch_api = (
            manual_maintaining_match["torch_api"].strip("`").replace(r"\_", "_")
        )
        torch_url = (
            manual_maintaining_match["torch_url"].strip("`").replace(r"\_", "_")
        )
        paddle_api = (
            manual_maintaining_match["paddle_api"]
            .strip("`")
            .replace(r"\_", "_")
        )
        paddle_url = (
            manual_maintaining_match["paddle_url"]
            .strip("`")
            .replace(r"\_", "_")
        )
        mapping_column = (
            manual_maintaining_match["mapping_type_desc"]
            .strip()
            .replace(r"\_", "_")
        )
        diff_page_url = (
            manual_maintaining_match["diff_url"].strip("`").replace(r"\_", "_")
        )
        mapping_url_column = f"[详细对比]({diff_page_url})"
        torch_api_column = f"[`{torch_api}`]({torch_url})"
        paddle_api_column = f"[`{paddle_api}`]({paddle_url})"
        content = [
            row_idx_s,
            torch_api_column,
            paddle_api_column,
            mapping_column,
            mapping_url_column,
        ]
        output = "| " + " | ".join(content) + " |\n"
        return [output]
    else:
        raise ValueError(
            f"found manual-maintaining row at line [{line_idx}]: {line}"
        )
        return [line]


def reference_mapping_item_processer(line, line_idx, state, output, context):
    if not line.startswith("|"):
        output.append(line)
        return True

    metadata_dict = context.get("metadata_dict", {})

    if state == 0:
        # check column names in common process
        output.append(line)
        return True
    elif state == 1 or state == 2:
        # check seperator of table to process in common process
        output.append(line)
        return True
    elif state == 5:
        # check content of table to ignore in common process
        output.append(line)
        return True
    elif state == 6:
        # check content of table to process in common process
        output_lines = apply_reference_to_row_ex(
            line, metadata_dict, context, line_idx + 1
        )

        output += output_lines
        return True

    print(state)
    return False


def reference_table_scanner(line, _line_idx, state, output, context):
    if not line.startswith("|"):
        return True

    if state >= 0 and state <= 2:
        return True
    elif state == 5:
        return True
    elif state == 6:
        # check content of table to process in common process
        rtm = REFERENCE_TABLE_PATTERN.match(line)
        if rtm:
            condition = reference_table_match_to_condition(rtm)
            context["table_conditions"].append(condition)
        return True

    return False


def get_c2a_dict(conditions, meta_dict):
    c2a_dict = {c: [] for c in conditions}
    conditions.sort(
        key=lambda c: (-len(c[0]), c[1])
    )  # 先按照字符串长度降序，随后按照最大深度升序
    for api in meta_dict:
        for api_prefix, max_depth in conditions:
            if not api.startswith(api_prefix):
                continue
            depth = len(api.split(".")) - 1
            if depth > max_depth:
                continue
            c2a_dict[(api_prefix, max_depth)].append(api)
            break
        else:
            print(f"Warning: cannot find a suitable condition for api {api}")

    return c2a_dict


if __name__ == "__main__":
    # convert from pytorch basedir
    cfp_basedir = os.path.dirname(__file__)
    # pytorch_api_mapping_cn
    mapping_index_file = os.path.join(cfp_basedir, "pytorch_api_mapping_cn.md")

    api_difference_basedir = os.path.join(cfp_basedir, "api_difference")

    mapping_file_pattern = re.compile(r"^torch\.(?P<api_name>.+)\.md$")
    # get all diff files (torch.*.md)
    diff_files = sorted(
        [
            os.path.join(path, filename)
            for path, _, file_list in os.walk(api_difference_basedir)
            for filename in file_list
            if mapping_file_pattern.match(filename)
        ]
    )

    metas = sorted(
        [get_meta_from_diff_file(f) for f in diff_files],
        key=lambda x: x["torch_api"],
    )

    meta_dict = {m["torch_api"].replace(r"\_", "_"): m for m in metas}

    reference_context = {
        "metadata_dict": meta_dict,
        "ret_code": 0,
        "output": [],
        "table_conditions": [],
    }

    # 第一遍预读，用来分析有哪些表格和匹配条件
    ret_code = reference_mapping_item(
        mapping_index_file, reference_table_scanner, reference_context
    )
    assert ret_code == 0
    reference_context["output"] = []

    # 现在 c2a_dict 包含每个条件对应的 api 列表
    c2a_dict = get_c2a_dict(reference_context["table_conditions"], meta_dict)
    reference_context["c2a_dict"] = c2a_dict

    # 第二遍正式读，读并处理
    ret_code = reference_mapping_item(
        mapping_index_file, reference_mapping_item_processer, reference_context
    )

    with open(mapping_index_file, "w", encoding="utf-8") as f:
        f.writelines(reference_context["output"])

    # 映射关系文件的保存流程移动至 `validate_mapping_in_api_difference.py`
