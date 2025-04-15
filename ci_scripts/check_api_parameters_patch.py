# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
import ast
import inspect
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import paddle  # noqa: F401

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Add Project Paths
THIS_DIR = Path(__file__).resolve().parent
API_DOC_TOOLS_PATH = THIS_DIR.parent / "docs" / "api"


def add_path(path: str):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(str(API_DOC_TOOLS_PATH))

from extract_api_from_docs import extract_params_desc_from_rst_file
from gen_doc import gen_functions_args_str

# Custom Exception


class APINotFoundError(Exception):
    pass


class APICheckError(Exception):
    pass


@dataclass
class CheckResults:
    passed: list[str] = field(default_factory=list)
    failed: dict[str] = field(default_factory=dict)
    not_found: dict[str] = field(default_factory=dict)


class ParamChecker:
    def __init__(self, api_info: dict[str, Any]):
        self.api_info = api_info
        self.api_info_by_name = {}
        for apiobj in api_info.values():
            if "all_names" in apiobj:
                for name in apiobj["all_names"]:
                    self.api_info_by_name[name] = apiobj

    def check_files(self, rst_files: list[Path]) -> CheckResults:
        results = CheckResults()
        for rst_file in rst_files:
            logger.info(f"Checking: {rst_file}")
            try:
                self.check_file(rst_file)
                results.passed.append(str(rst_file))
            except APINotFoundError as e:
                results.not_found[str(rst_file)] = str(e)
                logger.warning(f"API not found in {rst_file} - {e}")
            except APICheckError as e:
                results.failed[str(rst_file)] = str(e)
                logger.error(f"API check failed in {rst_file} - {e}")
            except Exception as e:
                results.failed[str(rst_file)] = str(e)
                logger.error(f"Unexpected error in {rst_file} - {e}")
        return results

    def check_file(self, rst_file: Path):
        pat = re.compile(
            r"^\.\.\s+py:(method|function|class)::\s+([^\s(]+)\s*(?:\(\s*(.*)\s*\))?\s*$"
        )
        with open(rst_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        func_found = False
        api_label = None

        for idx, line in enumerate(lines):
            line_strip = line.strip()
            if idx == 0:
                api_label = (
                    line_strip.removeprefix(".. _cn_api_")
                    .removesuffix(":")
                    .removesuffix("__upper")
                )
            mo = pat.match(line_strip)
            if mo:
                func_found = True
                functype, funcname, paramstr = mo.groups()
                if functype not in ("function", "method"):
                    return  #
                func_to_label = funcname.replace(".", "_")
                if api_label and func_to_label != api_label:
                    class_name = ".".join(funcname.split(".")[:-1])
                    class_to_label = class_name.replace(".", "_")
                    if class_to_label != api_label:
                        raise APICheckError(
                            f"Function name in title does not match the api label name: {funcname} != {api_label}"
                        )
                apiobj = self.api_info_by_name.get(funcname)
                if apiobj and "args" in apiobj:
                    if paramstr == apiobj["args"]:
                        self._check_params_in_description(rst_file, paramstr)
                    else:
                        logger.warning(
                            f"Parameter string mismatch for {funcname}: RST='{paramstr}', JSON='{apiobj['args']}'"
                        )
                        self._check_params_in_description(rst_file, paramstr)
                else:
                    self._check_params_in_description_with_fullargspec(
                        rst_file, funcname
                    )
                return  #
        if not func_found:
            raise APINotFoundError(
                "Function name in title not found, please check the format of '.. py:function::func()'"
            )

    def _check_params_in_description(
        self, rst_file: Path, paramstr: str | None
    ):
        params_in_title = []
        if paramstr:
            try:
                fake_func = ast.parse(f"def fake_func({paramstr}): pass")
                func_node = fake_func.body[0]
                func_args_str = gen_functions_args_str(func_node)
                params_in_title = [
                    p.split("=")[0].strip()
                    for p in func_args_str.split(", ")
                    if p not in ("/", "*")
                ]
                params_in_title = [
                    p.removeprefix("*").removeprefix("*")
                    for p in params_in_title
                ]
            except Exception as e:
                raise APICheckError(f"Failed to parse parameters: {e}")
        funcdescnode = extract_params_desc_from_rst_file(str(rst_file), True)
        if funcdescnode:
            try:
                items = funcdescnode.children[1].children[0].children
            except Exception:
                raise APICheckError(
                    "Params section format error in description."
                )
            if not items:
                if params_in_title:
                    raise APICheckError(
                        "Params section in description is empty, check it please."
                    )
            elif len(items) != len(params_in_title):
                raise APICheckError(
                    f"The number of params in title does not match the params in description: {len(params_in_title)} != {len(items)}."
                )
            else:
                for i, item in enumerate(items):
                    pname_in_title = params_in_title[i]
                    mo = re.match(
                        r"\*{0,2}(\w+)\b.*", item.children[0].astext()
                    )
                    if mo:
                        pname_indesc = mo.group(1)
                        if pname_indesc != pname_in_title:
                            raise APICheckError(
                                f"Param mismatch: {pname_in_title} != {pname_indesc}."
                            )
                    else:
                        raise APICheckError(
                            f"Param name '{pname_in_title}' not matched in description line {i + 1}, check it please."
                        )
        elif params_in_title:
            raise APICheckError(
                "Params section not found in description, check it please."
            )

    def _check_params_in_description_with_fullargspec(
        self, rst_file: Path, funcname: str
    ):
        try:
            obj = self._import_object(funcname)
        except Exception:
            raise APICheckError(
                f"Function {funcname} not found in paddle module, please check it."
            )
        try:
            source = inspect.getsource(obj)
            tree = ast.parse(source)
            func_node = tree.body[0]
            params_inspec = [
                p.split("=")[0].strip()
                for p in gen_functions_args_str(func_node).split(", ")
                if p not in ("/", "*")
            ]
            # for *args and **kwargs, remove * and **
            params_inspec = [
                p.removeprefix("*").removeprefix("*") for p in params_inspec
            ]

        except Exception as e:
            raise APICheckError(f"Failed to inspect function {funcname}: {e}")
        funcdescnode = extract_params_desc_from_rst_file(str(rst_file), True)
        if funcdescnode:
            try:
                items = funcdescnode.children[1].children[0].children
            except Exception:
                raise APICheckError(
                    "Params section format error in description."
                )
            if len(items) != len(params_inspec):
                raise APICheckError(
                    f"Param count mismatch: {len(params_inspec)} != {len(items)}."
                )
            else:
                for i, item in enumerate(items):
                    pname_in_title = params_inspec[i]
                    mo = re.match(
                        r"\*{0,2}(\w+)\b.*", item.children[0].astext()
                    )
                    if mo:
                        pname_indesc = mo.group(1)
                        if pname_indesc != pname_in_title:
                            raise APICheckError(
                                f"Param mismatch: {pname_in_title} != {pname_indesc}."
                            )
                    else:
                        raise APICheckError(
                            f"Param name '{pname_in_title}' not matched in description line {i + 1}."
                        )
        else:
            if params_inspec:
                raise APICheckError(
                    "Params section not found in description, check it please."
                )

    def _import_object(self, dotted_path: str):
        import importlib

        parts = dotted_path.split(".")
        module = importlib.import_module(parts[0])
        obj = module
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj


def parse_args():
    parser = argparse.ArgumentParser(description="check api parameters")
    parser.add_argument(
        "--rst-files",
        dest="rst_files",
        required=True,
        help="api rst files, separated by space",
        type=str,
    )
    parser.add_argument(
        "--api-info",
        dest="api_info_file",
        required=True,
        help="api_info_all.json filename",
        type=str,
    )
    parser.add_argument("--debug", dest="debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    try:
        with open(args.api_info_file, "r", encoding="utf-8") as f:
            api_info = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load API info file: {e}")
        sys.exit(1)
    rst_files = [fn for fn in args.rst_files.split(" ") if fn]
    if not rst_files:
        logger.error("No RST files provided.")
        sys.exit(1)

    checker = ParamChecker(api_info)
    results = checker.check_files(rst_files)

    logger.warning(
        f"API parameter checking completed. Pass: {len(results.passed)}, Fail: {len(results.failed)}, Not Found: {len(results.not_found)}"
    )

    if results.failed:
        logger.warning("Following files failed the check:")
        for file_path, error in results.failed.items():
            logger.error(f"  - {file_path}: {error}")
    if results.not_found:
        logger.warning("Following files had API not found:")
        for file_path, error in results.not_found.items():
            logger.error(f"  - {file_path}: {error}")
    if results.failed or results.not_found:
        sys.exit(1)
    else:
        logger.info("All API parameter checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
