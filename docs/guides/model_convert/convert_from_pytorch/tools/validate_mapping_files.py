from __future__ import annotations

import collections
import json
import os
import re
import sys
import traceback
import urllib
import urllib.parse
from enum import IntEnum
from typing import TypedDict

PADDLE_DOCS_BASE_URL = "https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/"

mapping_type_levels = [
    # type 0
    ["UNDEFINED_MAPPING_TYPE_0"],
    # type 1
    [
        "无参数",
        "参数完全一致",
        "仅参数名不一致",
        "paddle 参数更多",
        "参数默认值不一致",
    ],
    # type 2
    [
        "torch 参数更多",
    ],
    # type 3
    [
        "返回参数类型不一致",
        "输入参数类型不一致",
        "输入参数用法不一致",
    ],
    # type 4
    [
        "组合替代实现",
    ],
    # type 5
    [
        "涉及上下文修改",
    ],
    # type 6
    [
        "对应 API 不在主框架",
    ],
    # type 7
    [
        "功能缺失",
    ],
    # delete
    [
        "可删除",
    ],
]

mapping_type_set = {t for ts in mapping_type_levels for t in ts}
mapping_type_to_level = {
    t: level for level, ts in enumerate(mapping_type_levels) for t in ts
}


class DiffMeta(TypedDict):
    src_api: str
    src_api_url: str | None
    src_signature: str | None
    dst_api: str | None
    dst_api_url: str | None
    dst_signature: str | None
    args_mapping: list[dict[str, str]] | None
    mapping_type: str
    source_file: str


class IndexParserState(IntEnum):
    error = -1
    normal = 0
    table_sep = 1
    table_row = 2
    table_sep_ignore = 3
    table_row_ignore = 4


class ParserState(IntEnum):
    wait_for_title = 0

    wait_for_src_api = 1
    wait_for_src_signature_begin = 2
    wait_for_src_signature = 3
    wait_for_src_signature_end = 4

    wait_for_dst_api = 5
    wait_for_dst_signature_begin = 6
    wait_for_dst_signature = 7
    wait_for_dst_signature_end = 8

    wait_for_args = 9
    wait_for_args_table_title = 10
    wait_for_args_table_sep = 11
    wait_for_args_table_end = 12

    end = 13


def unescape_api(api):
    return api.replace(r"\_", "_")


def split_args(args_str):
    """
    按逗号分割参数字符串，忽略括号内的逗号
    """
    args = []
    current = []
    stack = []
    brackets = {"(": ")", "[": "]", "{": "}"}
    closing = {v: k for k, v in brackets.items()}

    for char in args_str:
        if char in brackets:
            stack.append(brackets[char])
        elif char in closing:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                raise ValueError(f"Unmatched closing bracket: {char}")
        elif char == "," and not stack:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue
        current.append(char)

    arg = "".join(current).strip()
    if arg:
        args.append(arg)
    return args


def split_signatures(buffer):
    """
    分割包含多个函数签名的字符串
    """
    signatures = []
    current = []
    stack = []
    brackets = {"(": ")", "[": "]", "{": "}"}
    closing = {v: k for k, v in brackets.items()}

    for char in buffer:
        if char in brackets:
            stack.append(brackets[char])
        elif char in closing:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                raise ValueError(f"Unmatched closing bracket: {char}")
        current.append(char)
        if char == ")" and not stack:
            # End of a function signature
            signature = "".join(current).strip()
            signatures.append(signature)
            current = []

    if current:
        signatures.append("".join(current).strip())

    return signatures


def reformat_signature(code):
    """
    从函数签名代码中解析出函数名和参数列表
    - code: 函数签名代码，可能包含多个函数声明
    - 返回值: 包含所有函数名和参数列表的列表
    """
    lines = [l for l in code.split("\n") if len(l.strip()) > 0]
    assert len(lines) > 0, "code have no lines."
    buffer = "".join([l.strip() for l in lines])

    signatures = split_signatures(buffer)

    parsed_signatures = []

    arg_pattern = re.compile(
        r"^(?P<arg_name>[^\=\:]+)"  # 参数名
        r"(?:\s*\:\s*(?P<arg_type>[^=]+))?"  # 可选的类型注解
        r"(?:\s*\=\s*(?P<arg_default>.+))?$"  # 可选的默认值
    )

    for sig in signatures:
        first_par_pos = sig.find("(")
        m = re.match(r"^\s*(?P<api_name>[^\( ]+)(.*?)$", sig)
        assert m is not None, f'code first line "{sig}" not match api pattern.'
        api_name = m.group("api_name")

        if first_par_pos < 0:
            # 要是没括号，可能是特殊情况，比如 property
            parsed_signatures.append({"api_name": api_name})
            continue

        last_par_pos = sig.rfind(")")
        assert (
            last_par_pos > first_par_pos
        ), f'code first line "{sig}" not match api pattern.'
        args_buffer = sig[first_par_pos + 1 : last_par_pos]

        args = []
        args_buffer = args_buffer.strip()

        arg_buffer_list = split_args(args_buffer)

        for arg_buffer in arg_buffer_list:
            m = arg_pattern.match(arg_buffer)
            assert (
                m is not None
            ), f'code arg "{arg_buffer}" not match arg pattern.'
            arg_name = m.group("arg_name").strip()
            arg_type = (
                m.group("arg_type").strip() if m.group("arg_type") else None
            )
            arg_default = (
                m.group("arg_default").strip()
                if m.group("arg_default")
                else None
            )

            if (
                arg_name.startswith("*")
                or arg_name.startswith("**")
                or arg_name[0].isalpha()
                or arg_name[0] == "_"
            ):
                # if is a valid arg name
                args.append(
                    {
                        "arg_name": arg_name,
                        "arg_type": arg_type,
                        "arg_default": arg_default,
                    }
                )
            else:
                if args:
                    if args[-1]["arg_default"] is not None:
                        args[-1]["arg_default"] += f", {arg_name}"
                    else:
                        args[-1]["arg_default"] = arg_name
                else:
                    raise ValueError(f"Invalid argument format: {arg_buffer}")

        parsed_signatures.append({"api_name": api_name, "args": args})

    return parsed_signatures


def get_meta_from_diff_file(
    filepath,
    src_prefix="torch.",
    dst_prefix="paddle.",
    src_argmap_title="PyTorch",
    dst_argmap_title="PaddlePaddle",
):
    """
    该函数从指定的映射文件中解析出元数据信息
    - filepath: 映射文件路径
    - src_prefix: 需要映射的 API 前缀
    - dst_prefix: 映射目标 API 前缀
    - 返回值: DiffMeta 类型的元数据信息
    """
    meta_data: DiffMeta = {"source_file": filepath}
    state = ParserState.wait_for_title
    title_pattern = re.compile(r"^## +\[(?P<type>[^\]]+)\] *(?P<src_api>.+)$")
    torch_pattern = re.compile(
        rf"^### +\[ *(?P<src_api>{re.escape(src_prefix)}[^\]]+)\](?P<url>\([^\)]*\))?$"
    )
    paddle_pattern = re.compile(
        rf"^### +\[ *(?P<dst_api>{re.escape(dst_prefix)}[^\]]+)\](\((?P<url>[^\)]*)\))?$"
    )
    code_begin_pattern = re.compile(r"^```python$")
    code_pattern = re.compile(r"^(?P<api_name>[^#][^\( ]+)(.*?)$")
    code_end_pattern = re.compile(r"^```$")

    args_pattern = re.compile(r"^### 参数映射$")
    ARGS_EXPECT_HEADERS = [src_argmap_title, dst_argmap_title, "备注"]

    mapping_type = ""
    signature_cache = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # 现在需要考虑内容信息了
            # if not line.startswith("##"):
            #     continue

            if state == ParserState.wait_for_title:
                title_match = title_pattern.match(line)
                if title_match:
                    mapping_type = title_match["type"].strip()
                    src_api = unescape_api(title_match["src_api"].strip())
                    if src_api == "torch.pi":
                        dst_prefix = "numpy."
                    meta_data["src_api"] = unescape_api(src_api)
                    meta_data["mapping_type"] = mapping_type

                    if mapping_type not in mapping_type_set:
                        raise ValueError(
                            f"Unexpected mapping type: {mapping_type} in {filepath}"
                        )

                    state = ParserState.wait_for_src_api
                else:
                    raise Exception(f"Cannot parse title: {line} in {filepath}")
            elif state == ParserState.wait_for_src_api:
                torch_match = torch_pattern.match(line)

                if torch_match:
                    src_api = torch_match["src_api"].strip()
                    torch_url = torch_match["url"] if torch_match["url"] else ""
                    real_url = torch_url.lstrip("(").rstrip(")")
                    if meta_data["src_api"] != unescape_api(src_api):
                        raise Exception(
                            f"torch api not match: {line} != {meta_data['src_api']} in {filepath}"
                        )
                    meta_data["src_api_url"] = real_url
                    state = ParserState.wait_for_src_signature_begin
            elif state == ParserState.wait_for_dst_api:
                paddle_pattern = re.compile(
                    rf"^### +\[ *(?P<dst_api>{re.escape(dst_prefix)}[^\]]+)\](\((?P<url>[^\)]*)\))?$"
                )
                paddle_match = paddle_pattern.match(line)

                if paddle_match:
                    dst_api = paddle_match["dst_api"].strip()
                    paddle_url = paddle_match["url"].strip()
                    meta_data["dst_api"] = unescape_api(dst_api)
                    meta_data["dst_api_url"] = paddle_url
                    state = ParserState.wait_for_dst_signature_begin
            elif state in [
                ParserState.wait_for_src_signature_begin,
                ParserState.wait_for_dst_signature_begin,
            ]:
                cb_match = code_begin_pattern.match(line)

                if cb_match:
                    if state == ParserState.wait_for_src_signature_begin:
                        state = ParserState.wait_for_src_signature
                    elif state == ParserState.wait_for_dst_signature_begin:
                        state = ParserState.wait_for_dst_signature
                    else:
                        raise ValueError(
                            f"Unexpected state {state} when process {filepath} line: {line}"
                        )
            elif state in [
                ParserState.wait_for_src_signature,
                ParserState.wait_for_dst_signature,
            ]:
                code_match = code_pattern.match(line)

                if code_match:
                    api_name = code_match["api_name"].strip()
                    if state == ParserState.wait_for_src_signature:
                        if api_name != meta_data["src_api"]:
                            raise ValueError(
                                f"Unexpected api code {api_name} != {meta_data['src_api']} when process {filepath} line: {line}"
                            )
                        else:
                            state = ParserState.wait_for_src_signature_end
                        signature_cache = line
                    elif state == ParserState.wait_for_dst_signature:
                        if api_name != meta_data["dst_api"]:
                            raise ValueError(
                                f"Unexpected api code {api_name} != {meta_data['dst_api']} when process {filepath} line: {line}"
                            )
                        else:
                            state = ParserState.wait_for_dst_signature_end
                        signature_cache = line
                else:
                    # 如果写注释，就先不管了
                    if line[0] != "#":
                        raise ValueError(
                            f"Api code must appear after ```, but not found correct signature when process {filepath} line: {line}."
                        )
            elif state in [
                ParserState.wait_for_src_signature_end,
                ParserState.wait_for_dst_signature_end,
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

                    if state == ParserState.wait_for_src_signature_end:
                        meta_data["src_signature"] = signature_info
                        state = ParserState.wait_for_dst_api
                    elif state == ParserState.wait_for_dst_signature_end:
                        meta_data["dst_signature"] = signature_info
                        state = ParserState.wait_for_args
                    else:
                        raise ValueError(
                            f"Unexpected state {state} when process {filepath} line: {line}"
                        )
                else:
                    # not match, append line to cache
                    if state == ParserState.wait_for_src_signature_end:
                        signature_cache += line
                    elif state == ParserState.wait_for_dst_signature_end:
                        signature_cache += line
                    else:
                        raise ValueError(
                            f"Unexpected state {state} when process {filepath} line: {line}"
                        )
            elif state == ParserState.wait_for_args:
                args_match = args_pattern.match(line)
                if args_match:
                    state = ParserState.wait_for_args_table_title
            elif state == ParserState.wait_for_args_table_title:
                if line.startswith("|"):
                    args_table_headers = [
                        c.strip() for c in line.split("|") if len(c.strip()) > 0
                    ]
                    if args_table_headers != ARGS_EXPECT_HEADERS:
                        raise Exception(
                            f"args mapping table headers mismatch, expect {ARGS_EXPECT_HEADERS}, but got {args_table_headers} in {filepath} ."
                        )
                    else:
                        state = ParserState.wait_for_args_table_sep
                        meta_data["args_mapping"] = []
            elif state == ParserState.wait_for_args_table_sep:
                if line.startswith("|"):
                    args_table_seps = [
                        c.strip() for c in line.split("|") if len(c.strip()) > 0
                    ]
                    if len(args_table_seps) == len(ARGS_EXPECT_HEADERS):
                        state = ParserState.wait_for_args_table_end
                    else:
                        raise Exception(
                            f"Unexpected args table seps: {args_table_seps} in {filepath}"
                        )
                else:
                    raise Exception(
                        f"Unexpected args table sep: {line} in {filepath}"
                    )
            elif state == ParserState.wait_for_args_table_end:
                if line.startswith("|"):
                    args_table_content = [c.strip() for c in line.split("|")][
                        1:-1
                    ]
                    if len(args_table_content) == len(ARGS_EXPECT_HEADERS):
                        torch_arg, paddle_arg, note = args_table_content
                        meta_data["args_mapping"].append(
                            {
                                "src_arg": torch_arg,
                                "dst_arg": paddle_arg,
                                "note": note,
                            }
                        )
                    else:
                        raise Exception(
                            f"Unexpected args table end: {args_table_content} in {filepath}"
                        )
                else:
                    state = ParserState.end
            elif state == ParserState.end:
                break
            else:
                raise ValueError(
                    f"Unexpected state {state} when process {filepath} line: {line}"
                )

    # 允许没有参数映射列表
    if mapping_type in ["无参数", "组合替代实现"]:
        if state == ParserState.wait_for_args:
            state = ParserState.end
    # 必须有参数映射列表，但是可以随时停止
    else:
        if state == ParserState.wait_for_args_table_end:
            state = ParserState.end

    if not mapping_type:
        raise Exception(
            f"Cannot get mapping_type in parsing file: {filepath}, current meta: {meta_data}"
        )

    # 允许的终止状态，解析完了 dst_api 或者只有 src_api
    # 映射类型前三个级别必须要有对应的 dst_api
    if mapping_type_to_level[mapping_type] <= 3:
        if state != ParserState.end:
            print(state)
            raise Exception(
                f"Unexpected End State at {state} in parsing file: {filepath}, current meta: {meta_data}"
            )
    else:
        if state not in [ParserState.end, ParserState.wait_for_dst_api]:
            raise Exception(
                f"Unexpected End State at {state} in parsing file: {filepath}, current meta: {meta_data}"
            )

    return meta_data


def process_mapping_index(index_path, item_processer, context={}):
    """
    线性处理 `pytorch_api_mapping_cn.md` 文件
    - index_path: 该 md 文件路径
    - item_processer: 对文件每行的处理方式，输入参数 (line, line_idx, state, output, context)。
                      如果处理出错则返回 False，否则返回 True。
    - context: 用于存储处理过程中的上下文信息
               - output: 使用 context["output"] 初始化，如果不调用 item_processer，直接加入原文件对应行，否则 item_processer 处理 output 逻辑。
    - 返回值：是否成功处理，成功返回 0。
    """
    if not os.path.exists(index_path):
        raise Exception(f"Cannot find pytorch_api_mapping_cn.md: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    state = IndexParserState.normal

    column_names = []
    column_count = -1
    table_seperator_pattern = re.compile(r"^ *\|(?P<group> *-+ *\|)+ *$")

    expect_column_names = [
        "序号",
        "Pytorch 最新 release",
        "Paddle develop",
        "映射关系分类",
        "备注",
    ]

    context["table_row_idx"] = context.get("table_row_idx", -1)
    output = context.get("output", [])

    for i, line in enumerate(lines):
        if state == IndexParserState.error:
            break

        content = line.strip()
        if not content.startswith("|"):
            output.append(line)
            state = IndexParserState.normal
            continue

        columns = [c.strip() for c in content.split("|")]
        columns = [c for c in columns if len(c) > 0]

        if state == IndexParserState.normal:
            column_names.clear()
            column_names.extend([c.strip() for c in columns])
            column_count = len(column_names)

            if not item_processer(line, i, state, output, context):
                break

            if column_names == expect_column_names:
                state = IndexParserState.table_sep
                context["table_row_idx"] = 1
                # print(f'process mapping table at line {i+1}.')
            else:
                state = IndexParserState.table_sep_ignore
                print(f"ignore table with {column_names} at line {i+1}.")

        elif state == IndexParserState.table_sep_ignore:
            if (
                not table_seperator_pattern.match(line)
                or len(columns) != column_count
            ):
                raise Exception(
                    f"Table seperator not match at line {i+1}: {line}"
                )
            if not item_processer(line, i, state, output, context):
                break
            state = IndexParserState.table_row_ignore
        elif state == IndexParserState.table_sep:
            if (
                not table_seperator_pattern.match(line)
                or len(columns) != column_count
            ):
                raise Exception(
                    f"Table seperator not match at line {i+1}: {line}"
                )
            if not item_processer(line, i, state, output, context):
                break
            state = IndexParserState.table_row
        elif state == IndexParserState.table_row_ignore:
            if not item_processer(line, i, state, output, context):
                break
        elif state == IndexParserState.table_row:
            try:
                context["columns"] = columns
                if not item_processer(line, i, state, output, context):
                    break
                context["table_row_idx"] += 1
            except Exception as e:
                print(e)
                print(f"Error at line {i+1}: {line}")
                traceback.print_exc()
                ret_code = 1
                sys.exit(-IndexParserState.table_row)
        else:
            ret_code = 2
            raise Exception(
                f"Unexpected State at {state} in processing file: {index_path}"
            )

    if state in [IndexParserState.table_row, IndexParserState.table_row_ignore]:
        state = IndexParserState.normal

    if state != IndexParserState.normal:
        raise Exception(
            f"Unexpected End State at {state} in parsing file: {index_path}"
        )

    ret_code = context.get("ret_code", 0xCC)
    if ret_code != 0:
        return ret_code

    context["output"] = output

    return 0


def get_doc_url_from_meta(basedir, meta):
    relpath = os.path.relpath(meta["source_file"], basedir).replace("\\", "/")
    diffurl = urllib.parse.urljoin(PADDLE_DOCS_BASE_URL, relpath)
    return diffurl


def generate_alias_lines_from_paconvert(basedir, meta_dict) -> None:
    alias_filename = "api_alias_mapping.json"
    alias_filepath = os.path.join(basedir, alias_filename)
    if not os.path.exists(alias_filepath) or not os.path.isfile(alias_filepath):
        return

    alias_refer_failed_list = []
    alias_output = {}
    with open(alias_filepath, "r", encoding="utf-8") as f:
        api_alias = json.load(f)
        for alias_name, api_name in api_alias.items():
            if api_name in meta_dict:
                pass
            elif alias_name in meta_dict:
                # 如果反着有，就交换
                api_name, alias_name = alias_name, api_name
            else:
                # 都没有就抛出警告
                alias_refer_failed_list.append((alias_name, api_name))
                continue

            meta_data = meta_dict[api_name]

            dst_api = meta_data.get("dst_api", "-")
            mapping_type = meta_data["mapping_type"]
            url = get_doc_url_from_meta(basedir, meta_data)

            alias_col = f"`{alias_name}`"
            paddle_col = f"`{dst_api}`"
            if "src_api_url" in meta_data:
                alias_col = f'[{alias_col}]({meta_data["src_api_url"]})'
            if "dst_api_url" in meta_data:
                paddle_col = f'[{paddle_col}]({meta_data["dst_api_url"]})'

            macro_line = f"ALIAS-REFERENCE-ITEM(`{alias_name}`, `{api_name}`)"
            alias_output[alias_name] = macro_line

    output_path = os.path.join(basedir, "alias_macro_lines.tmp.md")
    with open(output_path, "w", encoding="utf-8") as f:
        od_apis = collections.OrderedDict(sorted(alias_output.items()))
        for api, ref in od_apis.items():
            f.write(f"| {ref} |\n")

    print(f'generated alias temp file: "{output_path}"')

    if len(alias_refer_failed_list) > 0:
        fail_log_path = os.path.join(basedir, "alias_refer_failed.log")
        with open(fail_log_path, "w", encoding="utf-8") as f:
            for alias_name, api_name in alias_refer_failed_list:
                f.write(
                    f"api `{api_name}` have no mapping doc, failed to reference from alias `{alias_name}`\n"
                )

        print(
            f'{len(alias_refer_failed_list)} alias reference failed, see log file: "{fail_log_path}"'
        )


def discover_markdown_files(base_dir, prefix="torch."):
    pattern = re.compile(f"^{re.escape(prefix)}.*{re.escape('.md')}$")
    markdown_files = []
    for path, _, file_list in os.walk(base_dir):
        for filename in file_list:
            if pattern.match(filename):
                markdown_files.append(os.path.join(path, filename))
    return markdown_files


ARG_MAPPING_TABLE_HEADERS = {"torch.": "PyTorch"}


def get_table_header_by_prefix(prefix):
    if prefix in ARG_MAPPING_TABLE_HEADERS:
        return ARG_MAPPING_TABLE_HEADERS[prefix]
    assert prefix.endswith("."), f"prefix must end with '.' but got {prefix}"
    return prefix.rstrip(".")


def discover_all_metas(cfp_basedir):
    # 获取 api_difference/ 下的 api 映射文档
    diff_3rd_basedir = os.path.join(cfp_basedir, "api_difference_third_party")

    diff_srcs = [("api_difference", "torch.", "paddle.")]
    diff_srcs.extend(
        [
            (os.path.join(diff_3rd_basedir, subdir), f"{subdir}.", "")
            for subdir in os.listdir(diff_3rd_basedir)
        ]
    )

    diff_files = []
    for diff_src, api_prefix, dst_prefix in diff_srcs:
        basedir = os.path.join(cfp_basedir, diff_src)
        files = discover_markdown_files(basedir, api_prefix)
        diff_files.append(((api_prefix, dst_prefix), files))

        print(
            f"{len(files)} mapping documents found in {os.path.relpath(basedir, cfp_basedir)}."
        )

    metas = []
    for prefixs, files in diff_files:
        s, d = prefixs
        sh = get_table_header_by_prefix(s)
        for f in files:
            metas.append(get_meta_from_diff_file(f, s, d, src_argmap_title=sh))

    metas.sort(key=lambda x: x["src_api"])
    print(f"extracted {len(metas)} mapping metas data.")
    return metas


if __name__ == "__main__":
    # convert from pytorch basedir
    tools_dir = os.path.dirname(__file__)
    cfp_basedir = os.path.join(tools_dir, "..")
    # pysrc_api_mapping_cn
    mapping_index_file = os.path.join(cfp_basedir, "pytorch_api_mapping_cn.md")

    if not os.path.exists(mapping_index_file):
        raise Exception(f"Cannot find mapping index file: {mapping_index_file}")

    metas = discover_all_metas(cfp_basedir)

    for m in metas:
        if m["mapping_type"] not in mapping_type_set:
            print(m)
            raise Exception(
                f"Unknown mapping type: {m['mapping_type']} in {m['source_file']}"
            )

    meta_dict = {m["src_api"].replace(r"\_", "_"): m for m in metas}

    # 该文件用于 PaConvert 的文档对齐工作
    api_diff_output_path = os.path.join(tools_dir, "docs_mappings.json")

    with open(api_diff_output_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=4)

    generate_alias_lines_from_paconvert(tools_dir, meta_dict)
