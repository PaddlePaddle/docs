import os
import re
import sys

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)
print(script_dir)

from validate_mapping_in_api_difference import (
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


REFERENCE_PATTERN = re.compile(
    r"^\| *REFERENCE-MAPPING-ITEM\( *(?P<torch_api>[^,]+) *, *(?P<diff_url>.+) *\) *\|$"
)
ALIAS_PATTERN = re.compile(
    r"^\| *ALIAS-REFERENCE-ITEM\( *(?P<alias_name>[^,]+) *, *(?P<torch_api>[^,]+) *\) *\|$"
)
NOT_IMPLEMENTED_PATTERN = re.compile(
    r"^\| *NOT-IMPLEMENTED-ITEM\( *(?P<torch_api>[^,]+) *, *(?P<torch_api_url>.+) *\) *\|$"
)


DOCS_REPO_BASEURL = "https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/"


def docs_url_to_relative_page(url):
    if not url.startswith(DOCS_REPO_BASEURL):
        return url

    md_path = url[len(DOCS_REPO_BASEURL) :]
    if md_path.endswith(".md"):
        return md_path[:-3] + ".html"
    return md_path


def apply_reference_to_row(line, metadata_dict, table_row_idx, line_idx):
    reference_match = REFERENCE_PATTERN.match(line)
    alias_match = ALIAS_PATTERN.match(line)
    not_implemented_match = NOT_IMPLEMENTED_PATTERN.match(line)

    row_idx_s = str(table_row_idx)

    if reference_match:
        torch_api = reference_match["torch_api"].strip("`").replace(r"\_", "_")
        diff_url = reference_match["diff_url"]

        diff_page_url = docs_url_to_relative_page(diff_url)

        if torch_api not in metadata_dict:
            raise Exception(
                f"Cannot find torch_api: {torch_api} in line {line_idx}"
            )

        meta_dict[torch_api]["diff_url"] = diff_page_url

        reference_item = metadata_dict.get(torch_api, None)
        torch_api_url = reference_item["torch_api_url"]
        torch_api_column = f"[`{torch_api}`]({torch_api_url})"

        mapping_type = reference_item["mapping_type"]
        mapping_type_column = mapping_type

        _mapping_type_desc, show_diff_url = mapping_type_to_description(
            mapping_type
        )
        mapping_url_column = ""
        if show_diff_url:
            mapping_url_column = f"[详细对比]({diff_page_url})"

        if "paddle_api" not in reference_item:
            if mapping_type not in ["组合替代实现", "可删除", "功能缺失"]:
                print(
                    f"Cannot find paddle_api for torch_api: {torch_api} in line {line_idx}"
                )
            paddle_api_column = ""
        else:
            paddle_api = reference_item["paddle_api"]
            paddle_api_url = reference_item["paddle_api_url"]
            paddle_api_column = f"[`{paddle_api}`]({paddle_api_url})"

        content = [
            row_idx_s,
            torch_api_column,
            paddle_api_column,
            mapping_type_column,
            mapping_url_column,
        ]

        output = "| " + " | ".join(content) + " |\n"
        return output
    elif alias_match:
        alias_name = alias_match["alias_name"].strip("`").replace(r"\_", "_")
        torch_api = alias_match["torch_api"].strip("`").replace(r"\_", "_")

        if torch_api not in metadata_dict:
            raise Exception(
                f"Cannot find torch_api: {torch_api} in line {line_idx}"
            )

        diff_page_url = metadata_dict[torch_api].get("diff_url", "")

        reference_item = metadata_dict.get(torch_api, None)
        torch_api_url = reference_item["torch_api_url"]
        alisa_name_column = f"[`{alias_name}`]({torch_api_url})"

        mapping_type = reference_item["mapping_type"]
        mapping_type_column = mapping_type

        _mapping_type_desc, show_diff_url = mapping_type_to_description(
            mapping_type
        )

        desc_column = f"`{torch_api}` 别名"

        if show_diff_url:
            desc_column += f"，[详细对比]({diff_page_url})"

        if "paddle_api" not in reference_item:
            if mapping_type not in ["组合替代实现", "可删除", "功能缺失"]:
                print(
                    f"Cannot find paddle_api for torch_api: {torch_api} in line {line_idx}"
                )
            paddle_api_column = ""
        else:
            paddle_api = reference_item["paddle_api"]
            paddle_api_url = reference_item["paddle_api_url"]
            paddle_api_column = f"[`{paddle_api}`]({paddle_api_url})"

        content = [
            row_idx_s,
            alisa_name_column,
            paddle_api_column,
            mapping_type_column,
            desc_column,
        ]

        output = "| " + " | ".join(content) + " |\n"
        return output

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
        return output
    else:
        raise ValueError(
            f"found manual-maintaining row at line [{line_idx}]: {line}"
        )
        return line


def reference_mapping_item_processer(line, line_idx, state, output, context):
    if not line.startswith("|"):
        output.append(line)
        return True

    metadata_dict = context.get("metadata_dict", {})

    if state == 0:
        # check column names in common process
        output.append(line)
        return True
    elif state == 1:
        # check seperator of table to ignore in common process
        output.append(line)
        return True
    elif state == 2:
        # check seperator of table to process in common process
        output.append(line)
        return True
    elif state == 5:
        # check content of table to ignore in common process
        output.append(line)
        return True
    elif state == 6:
        # check content of table to process in common process
        referenced_row = apply_reference_to_row(
            line, metadata_dict, context["table_row_idx"], line_idx + 1
        )

        output.append(referenced_row)
        return True

    print(state)
    return False


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
    }
    ret_code = reference_mapping_item(
        mapping_index_file, reference_mapping_item_processer, reference_context
    )

    with open(mapping_index_file, "w", encoding="utf-8") as f:
        f.writelines(reference_context["output"])

    # 映射关系文件的保存流程移动至 `validate_mapping_in_api_difference.py`
