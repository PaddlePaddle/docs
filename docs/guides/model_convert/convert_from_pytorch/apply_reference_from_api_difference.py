import json
import os
import re
import typing


class DiffMeta(typing.TypedDict):
    torch_api: str
    torch_api_url: typing.Optional[str]
    paddle_api: typing.Optional[str]
    paddle_api_url: typing.Optional[str]
    mapping_type: str
    source_file: str


def get_meta_from_diff_file(filepath):
    meta_data: DiffMeta = {"source_file": filepath}
    state = 0
    # 0: wait for title
    # 1: wait for torch api
    # 2: wait for paddle api
    # 3: end
    title_pattern = re.compile(r"^## +\[(?P<type>[^\]]+)\] *(?P<torch_api>.+)$")
    torch_pattern = re.compile(
        r"^### +\[ *(?P<torch_api>torch.[^\]]+)\](?P<url>\([^\)]*\))?$"
    )
    paddle_pattern = re.compile(
        r"^### +\[ *(?P<paddle_api>paddle.[^\]]+)\](?P<url>\([^\)]*\))$"
    )

    with open(filepath, "r") as f:
        for line in f.readlines():
            if not line.startswith("##"):
                continue

            if state == 0:
                title_match = title_pattern.match(line)
                if title_match:
                    mapping_type = title_match["type"].strip()
                    torch_api = title_match["torch_api"].strip()

                    meta_data["torch_api"] = torch_api
                    meta_data["mapping_type"] = mapping_type
                    state = 1
                else:
                    raise Exception(f"Cannot parse title: {line} in {filepath}")
            elif state == 1:
                torch_match = torch_pattern.match(line)

                if torch_match:
                    torch_api = torch_match["torch_api"].strip()
                    torch_url = torch_match["url"] if torch_match["url"] else ""
                    real_url = torch_url.lstrip("(").rstrip(")")
                    if meta_data["torch_api"] != torch_api:
                        raise Exception(
                            f"torch api not match: {line} != {meta_data['torch_api']} in {filepath}"
                        )
                    meta_data["torch_api_url"] = real_url
                    state = 2
                else:
                    raise Exception(
                        f"Cannot parse torch api: {line} in {filepath}"
                    )
            elif state == 2:
                paddle_match = paddle_pattern.match(line)

                if paddle_match:
                    paddle_api = paddle_match["paddle_api"].strip()
                    paddle_url = paddle_match["url"].strip()
                    real_url = paddle_url.lstrip("(").rstrip(")")
                    meta_data["paddle_api"] = paddle_api
                    meta_data["paddle_api_url"] = real_url
                    state = 3
            else:
                pass

    if state < 2:
        raise Exception(
            f"Unexpected End State at {state} in parsing file: {filepath}, current meta: {meta_data}"
        )

    return meta_data


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
        "返回参数类型不一致",
        "参数不一致",
        "参数用法不一致",
    ]
    if mapping_type in mapping_type_3:
        return "功能一致，" + mapping_type, True

    mapping_type_4 = ["组合替代实现"]
    if mapping_type in mapping_type_4:
        return "组合替代实现", True

    mapping_type_5 = ["用法不同：涉及上下文修改"]
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

    return "【未知类型】", False


REFERENCE_PATTERN = re.compile(
    r"^\| *REFERENCE-MAPPING-ITEM\( *(?P<torch_api>[^,]+) *, *(?P<diff_url>.+) *\) *\|$"
)
NOT_IMPLEMENTED_PATTERN = re.compile(
    r"^\| *NOT-IMPLEMENTED-ITEM\( *(?P<torch_api>[^,]+) *, *(?P<torch_api_url>.+) *\) *\|$"
)


def apply_reference_to_row(line, metadata_dict, table_row_idx, line_idx):
    reference_match = REFERENCE_PATTERN.match(line)
    not_implemented_match = NOT_IMPLEMENTED_PATTERN.match(line)

    if reference_match:
        torch_api = reference_match["torch_api"].strip("`").replace(r"\_", "_")
        diff_url = reference_match["diff_url"]

        row_idx_s = str(table_row_idx)

        if torch_api not in metadata_dict:
            raise Exception(
                f"Cannot find torch_api: {torch_api} in line {line_idx}"
            )

        reference_item = metadata_dict.get(torch_api, None)
        torch_api_url = reference_item["torch_api_url"]
        torch_api_column = f"[`{torch_api}`]({torch_api_url})"

        mapping_type = reference_item["mapping_type"]
        mapping_type_s, show_diff_url = mapping_type_to_description(
            mapping_type
        )
        mapping_column = mapping_type_s
        if show_diff_url:
            mapping_column += f"，[差异对比]({diff_url})"

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
            mapping_column,
        ]

        output = "| " + " | ".join(content) + " |\n"
        return output
    elif not_implemented_match:
        torch_api = (
            not_implemented_match["torch_api"].strip("`").replace(r"\_", "_")
        )
        torch_api_url = not_implemented_match["torch_api_url"].strip()

        row_idx_s = str(table_row_idx)

        torch_api_column = f"[`{torch_api}`]({torch_api_url})"

        paddle_api_column = ""
        mapping_column = "功能缺失"

        content = [
            row_idx_s,
            torch_api_column,
            paddle_api_column,
            mapping_column,
        ]

        output = "| " + " | ".join(content) + " |\n"
        return output
    else:
        print(f"found manual-maintaining row at line [{line_idx}]: {line}")
        return line


def reference_mapping_item(index_path, metadata_dict):
    if not os.path.exists(index_path):
        raise Exception(f"Cannot find pytorch_api_mapping_cn.md: {index_path}")

    with open(mapping_index_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    state = 0
    # -1: error
    # 0: wait for table header

    # 1: wait for ignore table seperator
    # 2: wait for expect table content

    # 5: wait for ignore table content
    # 6: wait for expect table content

    column_names = []
    column_count = -1
    table_seperator_pattern = re.compile(r"^ *\|(?P<group> *-+ *\|)+ *$")

    expect_column_names = ["序号", "PyTorch API", "PaddlePaddle API", "备注"]

    table_row_idx = -1
    output = []

    for i, line in enumerate(lines):
        if state < 0:
            break

        content = line.strip()
        if not content.startswith("|"):
            output.append(line)
            state = 0
            continue

        columns = [c.strip() for c in content.split("|")]
        if len(columns) <= 2:
            raise Exception(
                f"Table column count must > 0, but found {len(columns) - 2} at line {i+1}: {line}"
            )
        columns = columns[1:-1]

        if state == 0:
            column_names.clear()
            column_names.extend([c.strip() for c in columns])
            column_count = len(column_names)
            if column_names == expect_column_names:
                state = 2
                table_row_idx = 1
                # print(f'process mapping table at line {i+1}.')
            else:
                state = 1
                print(f"ignore table with {column_names} at line {i+1}.")
            output.append(line)
        elif state == 1:
            if (
                not table_seperator_pattern.match(line)
                or len(columns) != column_count
            ):
                raise Exception(
                    f"Table seperator not match at line {i+1}: {line}"
                )
            state = 5
            output.append(line)
        elif state == 2:
            if (
                not table_seperator_pattern.match(line)
                or len(columns) != column_count
            ):
                raise Exception(
                    f"Table seperator not match at line {i+1}: {line}"
                )
            state = 6
            output.append(line)
        elif state == 5:
            # if len(columns) != column_count:
            #     raise Exception(
            #         f"Table content not match at line {i+1}: {line}"
            #     )
            output.append(line)
            # state = 5
        elif state == 6:
            # if len(columns) != column_count:
            #     raise Exception(
            #         f"Table content not match at line {i+1}: {line}"
            #     )
            try:
                referenced_row = apply_reference_to_row(
                    line, metadata_dict, table_row_idx, i + 1
                )
                table_row_idx += 1

                output.append(referenced_row)
            except Exception as e:
                print(e)
                print(f"Error at line {i+1}: {line}")
                output.append(line)

            # state = 6
        else:
            raise Exception(
                f"Unexpected State at {state} in processing file: {index_path}"
            )

    if state == 5 or state == 6:
        state = 0

    if state != 0:
        raise Exception(
            f"Unexpected End State at {state} in parsing file: {index_path}"
        )

    with open(mapping_index_file, "w", encoding="utf-8") as f:
        f.writelines(output)


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

    reference_mapping_item(mapping_index_file, meta_dict)

    api_diff_output_path = os.path.join(cfp_basedir, "api_mappings.json")

    with open(api_diff_output_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=4)
