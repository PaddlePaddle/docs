import sys
import os
import re
import logging
import argparse
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
def check_api_label(rootdir, file):
    real_file = Path(rootdir) / file
    with open(real_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    print(first_line)
    print(generate_en_label_by_path(file))
    return first_line == generate_en_label_by_path(file)


# path -> api_label (the first line's style)
def generate_en_label_by_path(file):
    result = file.removeprefix(API)
    result = result.removesuffix('_cn.rst')
    result = result.replace('/', '_')
    result = f'.. _cn_{result}:'
    return result


# traverse doc/api to append api_label in list
def find_all_api_labels_in_dir(rootdir):
    all_api_labels = []
    for root, dirs, files in os.walk(rootdir + API):
        for file in files:
            real_path = Path(root) / file
            path = str(real_path).removeprefix(rootdir)
            if not should_test(path):
                continue
            for label in find_api_labels_in_one_file(real_path):
                all_api_labels.append(label)
    return all_api_labels


# api_labels in a file
def find_api_labels_in_one_file(file_path):
    api_labels_in_one_file = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = re.search(".. _([a-zA-Z0-9_]+)", line)
            if not line:
                continue
            api_labels_in_one_file.append(line.group(1))
    return api_labels_in_one_file


# api doc for checking
def should_test(file):
    return (
        file.endswith("_cn.rst")
        and not file.endswith("Overview_cn.rst")
        and not file.endswith("index_cn.rst")
        and file.startswith(API)
    )


def pipline(rootdir, files):
    for file in files:
        if should_test(file) and not check_api_label(rootdir, file):
            logger.error(
                f"The first line in {rootdir}/{file} is not avaiable, please re-check it!"
            )
            sys.exit(1)
    valid_api_labels = find_all_api_labels_in_dir(rootdir)
    for file in files:
        with open(Path(rootdir) / file, 'r', encoding='utf-8') as f:
            pattern = f.read()
        matches = re.findall(r":ref:`([^`]+)`", pattern)
        for match in matches:
            api_label = match
            if api_label_match := re.math(r".+<(?P<api_label>.+?)>", api_label):
                api_label = api_label_match.group("api_label")
            if api_label.startwith('cn_') and api_label not in valid_api_labels:
                logger.error(
                    f"Found api label {api_label} in {rootdir}/{file}, but it is not a valid api label, please re-check it!"
                )
                sys.exit(1)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='cn api_label checking')
    parser.add_argument(
        'rootdir',
        help='the dir DOCROOT',
        type=str,
        default='/FluidDoc/docs/',
    )

    parser.add_argument(
        'apiroot',
        type=str,
        help='the dir APIROOT',
        default='/FluidDoc/docs/api/',
    )
    parser.add_argument(
        'all_git_files',
        type=str,
        nargs='+',
        help='files need to check',
        default='',
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    API = args.apiroot.removeprefix(args.rootdir + '/')
    pipline(args.rootdir, args.all_git_files)
