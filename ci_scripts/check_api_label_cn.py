from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger()
if logger.handlers:
    # we assume the first handler is the one we want to configure
    console = logger.handlers[0]
else:
    console = logging.StreamHandler()
    logger.addHandler(console)
console.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
    )
)
logger.setLevel(logging.INFO)


# check file's api_label
def check_api_label(doc_root: str, file: str) -> bool:
    real_file = Path(doc_root) / file
    with open(real_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    return first_line == generate_cn_label_by_path(file)


# path -> api_label (the first line's style)
def generate_cn_label_by_path(file: str) -> str:
    result = file.removesuffix("_cn.rst")
    result = "_".join(Path(result).parts)
    result = f".. _cn_{result}:"
    return result


# traverse doc/api to append api_label in list
def find_all_api_labels_in_dir(api_root: str) -> list[str]:
    all_api_labels = []

    for file_path in Path(api_root).rglob("*.rst"):
        if not file_path.is_file():
            continue
        path = str(file_path).removeprefix(api_root.removesuffix(API))
        if not need_check(path):
            continue
        for label in find_api_labels_in_file(file_path):
            all_api_labels.append(label)
    return all_api_labels


# api_labels in a file
def find_api_labels_in_file(file_path: Path | str) -> list[str]:
    api_labels_in_one_file = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = re.search(".. _cn_api_paddle_([a-zA-Z0-9_]+)", line)
            if not line:
                continue
            api_labels_in_one_file.append(line.group(1))
    return api_labels_in_one_file


# api doc for checking
def need_check(file: str) -> bool:
    return (
        file.endswith("_cn.rst")
        and not Path(file).name == "Overview_cn.rst"
        and not Path(file).name == "index_cn.rst"
        and file.startswith(API)
    )


def check_usage_of_api_label(
    files: list[Path], valid_api_labels: list[str]
) -> list[str]:
    errors = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            pattern = f.read()
        matches = re.findall(r":ref:`([^`]+)`", pattern)
        for match in matches:
            api_label = match
            if api_label_match := re.match(
                r".+<(?P<api_label>.+?)>", api_label
            ):
                api_label = api_label_match.group("api_label")
            if not api_label.startswith("cn_api_paddle"):
                continue
            if api_label in valid_api_labels:
                continue
            errors.append(f"api label `{api_label}` in `{file}`")
    return errors


def get_custom_files_for_checking_usage(doc_root: str) -> set[Path]:
    # TODO: add more dir for checking
    custom_files = set()
    for file_path in (Path(doc_root) / API).rglob("*.rst"):
        if not file_path.is_file():
            continue
        custom_files.add(file_path)
    return custom_files


def run_cn_api_label_checking(
    doc_root: str, api_root: str, files: list[str]
) -> None:
    # check the api_label in the first line for increased files
    for file in files:
        if need_check(file) and not check_api_label(doc_root, file):
            logger.error(
                f"The first line in {doc_root}/{file} is not available, please re-check it!"
            )
            sys.exit(1)

    # collect all api_labels in api_root
    valid_api_labels = find_all_api_labels_in_dir(api_root)

    # check the usage of api_label in custom files
    api_label_usage_file_set = {Path(doc_root) / file for file in files}
    api_label_usage_file_set.update(
        get_custom_files_for_checking_usage(doc_root)
    )

    errors = check_usage_of_api_label(
        api_label_usage_file_set, valid_api_labels
    )
    if errors:
        logger.error("Found valid api labels usage as follows:")
        for i, error in enumerate(errors):
            logger.error(f"{i + 1}: {error}")
        sys.exit(1)

    print("All api_label check success in PR !")


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="cn api_label checking")
    parser.add_argument(
        "doc_root",
        help="the dir DOCROOT",
        type=str,
        default="/FluidDoc/docs/",
    )

    parser.add_argument(
        "api_root",
        type=str,
        help="the dir api_root",
        default="/FluidDoc/docs/api/",
    )
    parser.add_argument(
        "all_git_files",
        type=str,
        nargs="*",
        help="files need to check",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    API = args.doc_root.removesuffix(args.api_root)
    run_cn_api_label_checking(args.doc_root, args.api_root, args.all_git_files)
