import json
import os
import re
import typing
from enum import IntEnum
from typing import TypedDict

mapping_type_set = {
    # type 1
    "无参数",
    "参数完全一致",
    "仅参数名不一致",
    "仅 paddle 参数更多",
    "仅参数默认值不一致",
    # type 2
    "torch 参数更多",
    # type 3
    "返回参数类型不一致",
    "参数不一致",
    "参数用法不一致",
    # type 4
    "组合替代实现",
    # type 5
    "用法不同：涉及上下文修改",
    # type 6
    "对应 API 不在主框架",
    # type 7
    "功能缺失",
    # delete
    "可删除",
}


class DiffMeta(TypedDict):
    torch_api: str
    torch_api_url: typing.Optional[str]
    torch_signature: typing.Optional[str]
    paddle_api: typing.Optional[str]
    paddle_api_url: typing.Optional[str]
    paddle_signature: typing.Optional[str]
    mapping_type: str
    source_file: str


class ParserState(IntEnum):
    wait_for_title = 0

    wait_for_torch_api = 1
    wf_torch_code_begin = 2
    wf_torch_code = 3
    wf_torch_code_end = 4

    wait_for_paddle_api = 5
    wf_paddle_code_begin = 6
    wf_paddle_code = 7
    wf_paddle_code_end = 8

    end = 9


def unescape_api(api):
    return api.replace(r"\_", "_")


def reformat_signature(code):
    lines = [l for l in code.split("\n") if len(l.strip()) > 0]
    assert len(lines) > 0, "code have no lines."
    buffer = "".join([l.strip() for l in lines])

    first_par_pos = buffer.find("(")

    m = re.match(r"^\s*(?P<api_name>[^\( ]+)(.*?)$", buffer)
    assert m is not None, f'code first line "{buffer}" not match api pattern.'
    api_name = m.group("api_name")

    if first_par_pos < 0:
        # 要是没括号，可能是特殊情况，比如 property
        return {"api_name": api_name}

    last_par_pos = buffer.rfind(")")
    assert (
        last_par_pos > first_par_pos
    ), f'code first line "{buffer}" not match api pattern.'
    args_buffer = buffer[first_par_pos + 1 : last_par_pos]

    args = []
    args_buffer = args_buffer.strip()
    arg_pattern = re.compile(
        r"^(?P<arg_name>[^\=]+)(\=(?P<arg_default>[^,]+))?$"
    )

    arg_buffer_list = [
        l.strip() for l in args_buffer.split(",") if len(l.strip()) > 0
    ]
    for arg_buffer in arg_buffer_list:
        m = arg_pattern.match(arg_buffer)
        assert m is not None, f'code arg "{arg_buffer}" not match arg pattern.'
        arg_name = m.group("arg_name")
        arg_default = m.group("arg_default")
        args.append({"arg_name": arg_name, "arg_default": arg_default})

    return {"api_name": api_name, "args": args}


def get_meta_from_diff_file(filepath):
    meta_data: DiffMeta = {"source_file": filepath}
    state = ParserState.wait_for_title
    title_pattern = re.compile(r"^## +\[(?P<type>[^\]]+)\] *(?P<torch_api>.+)$")
    torch_pattern = re.compile(
        r"^### +\[ *(?P<torch_api>torch.[^\]]+)\](?P<url>\([^\)]*\))?$"
    )
    paddle_pattern = re.compile(
        r"^### +\[ *(?P<paddle_api>paddle.[^\]]+)\](?P<url>\([^\)]*\))?$"
    )
    code_begin_pattern = re.compile(r"^```(python)?$")
    code_pattern = re.compile(r"^(?P<api_name>(paddle|torch)[^\( ]+)(.*?)$")
    code_end_pattern = re.compile(r"^```$")

    signature_cache = None

    with open(filepath, "r") as f:
        for line in f.readlines():
            # 现在需要考虑内容信息了
            # if not line.startswith("##"):
            #     continue

            if state == ParserState.wait_for_title:
                title_match = title_pattern.match(line)
                if title_match:
                    mapping_type = title_match["type"].strip()
                    torch_api = title_match["torch_api"].strip()

                    meta_data["torch_api"] = unescape_api(torch_api)
                    meta_data["mapping_type"] = mapping_type
                    state = ParserState.wait_for_torch_api
                else:
                    raise Exception(f"Cannot parse title: {line} in {filepath}")
            elif state == ParserState.wait_for_torch_api:
                torch_match = torch_pattern.match(line)

                if torch_match:
                    torch_api = torch_match["torch_api"].strip()
                    torch_url = torch_match["url"] if torch_match["url"] else ""
                    real_url = torch_url.lstrip("(").rstrip(")")
                    if meta_data["torch_api"] != unescape_api(torch_api):
                        raise Exception(
                            f"torch api not match: {line} != {meta_data['torch_api']} in {filepath}"
                        )
                    meta_data["torch_api_url"] = real_url
                    state = ParserState.wf_torch_code_begin
            elif state == ParserState.wait_for_paddle_api:
                paddle_match = paddle_pattern.match(line)

                if paddle_match:
                    paddle_api = paddle_match["paddle_api"].strip()
                    paddle_url = paddle_match["url"].strip()
                    meta_data["paddle_api"] = unescape_api(paddle_api)
                    meta_data["paddle_api_url"] = paddle_url
                    state = ParserState.wf_paddle_code_begin
            elif state in [
                ParserState.wf_torch_code_begin,
                ParserState.wf_paddle_code_begin,
            ]:
                cb_match = code_begin_pattern.match(line)

                if cb_match:
                    if state == ParserState.wf_torch_code_begin:
                        state = ParserState.wf_torch_code
                    elif state == ParserState.wf_paddle_code_begin:
                        state = ParserState.wf_paddle_code
                    else:
                        raise ValueError(
                            f"Unexpected state {state} when process {filepath} line: {line}"
                        )
            elif state in [
                ParserState.wf_torch_code,
                ParserState.wf_paddle_code,
            ]:
                code_match = code_pattern.match(line)

                if code_match:
                    api_name = code_match["api_name"].strip()
                    if state == ParserState.wf_torch_code:
                        if api_name != meta_data["torch_api"]:
                            raise ValueError(
                                f"Unexpected api code {api_name} != {meta_data['torch_api']} when process {filepath} line: {line}"
                            )
                        else:
                            state = ParserState.wf_torch_code_end
                        signature_cache = line
                    elif state == ParserState.wf_paddle_code:
                        if api_name != meta_data["paddle_api"]:
                            raise ValueError(
                                f"Unexpected api code {api_name} != {meta_data['paddle_api']} when process {filepath} line: {line}"
                            )
                        else:
                            state = ParserState.wf_paddle_code_end
                        signature_cache = line
                else:
                    # 如果写注释，就先不管了
                    if line[0] != "#":
                        raise ValueError(
                            f"Api code must appear after ```, but not found correct signature when process {filepath} line: {line}."
                        )
            elif state in [
                ParserState.wf_torch_code_end,
                ParserState.wf_paddle_code_end,
            ]:
                ce_match = code_end_pattern.match(line)

                if ce_match:
                    try:
                        signature_info = reformat_signature(signature_cache)
                        signature_cache = None
                    except AssertionError as e:
                        raise Exception(
                            f"Cannot parse signature code in {filepath}"
                        ) from e

                    if state == ParserState.wf_torch_code_end:
                        meta_data["torch_signature"] = signature_info
                        state = ParserState.wait_for_paddle_api
                    elif state == ParserState.wf_paddle_code_end:
                        meta_data["paddle_signature"] = signature_info
                        state = ParserState.end
                    else:
                        raise ValueError(
                            f"Unexpected state {state} when process {filepath} line: {line}"
                        )
                else:
                    # not match, append line to cache
                    if state == ParserState.wf_torch_code_end:
                        signature_cache += line
                    elif state == ParserState.wf_paddle_code_end:
                        signature_cache += line
                    else:
                        raise ValueError(
                            f"Unexpected state {state} when process {filepath} line: {line}"
                        )
            elif state == ParserState.end:
                break
            else:
                raise ValueError(
                    f"Unexpected state {state} when process {filepath} line: {line}"
                )

    # print(state)

    # 允许的终止状态，解析完了 paddle_api 或者只有 torch_api
    if state not in [ParserState.end, ParserState.wait_for_paddle_api]:
        raise Exception(
            f"Unexpected End State at {state} in parsing file: {filepath}, current meta: {meta_data}"
        )

    return meta_data


# torch api must starts with "torch."
TABLE_COLUMN_TORCH_API_PATTERN = re.compile(
    r"^\[ *(?P<torch_api>torch\.[^\]]+) *\](?P<url>\([^\)]*\))$"
)

# paddle api must starts with "paddle"
TABLE_COLUMN_PADDLE_API_PATTERN = re.compile(
    r"^\[ *(?P<paddle_api>paddle[^\]]+) *\](?P<url>\([^\)]*\))$"
)

TABLE_COLUMN_MAPPING_PATTERN = re.compile(
    r"^(?P<type>[^\[]*)(\[(?P<diff_name>[^\]]+)\]\((?P<diff_url>[^\)]+)\))?"
)

MAPPING_DIFF_SOURCE_PATTERN = re.compile(
    r"^https://github.com/PaddlePaddle/((docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/)|(X2Paddle/tree/develop/docs/pytorch_project_convertor/API_docs/))"
)


def validate_mapping_table_row(columns, row_idx, line_idx):
    idx_s, torch_api_s, paddle_api_s, mapping_s = columns

    idx = int(idx_s)
    if row_idx != idx:
        raise Exception(
            f"Table row index [{row_idx}] != {idx} at line {line_idx}."
        )

    torch_api_match = TABLE_COLUMN_TORCH_API_PATTERN.match(torch_api_s)
    if torch_api_match:
        torch_api = torch_api_match["torch_api"]
        torch_api_url = torch_api_match["url"][1:-1]  # remove '(' and ')'
    else:
        raise Exception(
            f"Table row torch api not match: {torch_api_s} at line {line_idx}."
        )

    paddle_api_match = TABLE_COLUMN_PADDLE_API_PATTERN.match(paddle_api_s)
    if len(paddle_api_s) > 0:
        if paddle_api_match:
            paddle_api = paddle_api_match["paddle_api"]
            paddle_api_url = paddle_api_match["url"][1:-1]  # remove '(' and ')'
        else:
            raise Exception(
                f"Table row paddle api not match: {paddle_api_s} at line {line_idx}."
            )
    else:
        paddle_api = None
        paddle_api_url = None

    mapping_type_match = TABLE_COLUMN_MAPPING_PATTERN.match(mapping_s)
    if mapping_type_match:
        mapping_type = mapping_type_match["type"].strip()
        mapping_diff_name = mapping_type_match["diff_name"]
        diff_url = mapping_type_match["diff_url"]

        if mapping_diff_name != "差异对比" and mapping_diff_name is not None:
            print(
                f"Table row mapping diff name not match: {mapping_diff_name} at line {line_idx}."
            )

        if diff_url is not None and not MAPPING_DIFF_SOURCE_PATTERN.match(
            diff_url
        ):
            raise Exception(
                f"Table row mapping diff url invalid: {diff_url} at line {line_idx}."
            )
        mapping_diff_url = diff_url
    else:
        raise Exception(
            f"Table row mapping type not match: {mapping_s} at line {line_idx}."
        )

    return {
        "torch_api": torch_api,
        "torch_api_url": torch_api_url,
        "paddle_api": paddle_api,
        "paddle_api_url": paddle_api_url,
        "mapping_type": mapping_type,
        "mapping_diff_name": mapping_diff_name,
        "mapping_diff_url": mapping_diff_url,
        "line_idx": line_idx,
    }


def process_mapping_index(filename):
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

    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            if state < 0:
                break

            content = line.strip()
            if len(content) == 0 or content[0] != "|":
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
            elif state == 1:
                if (
                    not table_seperator_pattern.match(line)
                    or len(columns) != column_count
                ):
                    raise Exception(
                        f"Table seperator not match at line {i+1}: {line}"
                    )
                state = 5
            elif state == 2:
                if (
                    not table_seperator_pattern.match(line)
                    or len(columns) != column_count
                ):
                    raise Exception(
                        f"Table seperator not match at line {i+1}: {line}"
                    )
                state = 6
            elif state == 5:
                if len(columns) != column_count:
                    raise Exception(
                        f"Table content not match at line {i+1}: {line}"
                    )
                # state = 5
            elif state == 6:
                if len(columns) != column_count:
                    raise Exception(
                        f"Table content not match at line {i+1}: {line}"
                    )

                item = validate_mapping_table_row(columns, table_row_idx, i + 1)
                table_row_idx += 1

                output.append(item)

                # state = 6
            else:
                raise Exception(
                    f"Unexpected State at {state} in parsing file: {filename}"
                )

    if state == 5 or state == 6:
        state = 0

    if state != 0:
        raise Exception(
            f"Unexpected End State at {state} in parsing file: {filename}"
        )

    return output


if __name__ == "__main__":
    # convert from pytorch basedir
    cfp_basedir = os.path.dirname(__file__)
    # pytorch_api_mapping_cn
    mapping_index_file = os.path.join(cfp_basedir, "pytorch_api_mapping_cn.md")

    if not os.path.exists(mapping_index_file):
        raise Exception(f"Cannot find mapping index file: {mapping_index_file}")

    # index_data = process_mapping_index(mapping_index_file)
    # index_data_dict = {i['torch_api'].replace('\_', '_'): i for i in index_data}

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
    print(f"{len(diff_files)} mapping documents found.")

    metas = sorted(
        [get_meta_from_diff_file(f) for f in diff_files],
        key=lambda x: x["torch_api"],
    )
    print(f"extracted {len(metas)} mapping metas data.")

    for m in metas:
        if m["mapping_type"] not in mapping_type_set:
            print(m)
            raise Exception(
                f"Unknown mapping type: {m['mapping_type']} in {m['source_file']}"
            )

    meta_dict = {m["torch_api"].replace(r"\_", "_"): m for m in metas}

    # 该文件用于 PaConvert 的文档对齐工作
    api_diff_output_path = os.path.join(cfp_basedir, "docs_mappings.json")

    with open(api_diff_output_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=4)
