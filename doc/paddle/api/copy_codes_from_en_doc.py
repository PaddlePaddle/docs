#! /bin/env python
'''
copy code-blocks from en api doc-strings.
'''
import os
import sys
import argparse


def instert_codes_into_cn_rst_if_need(cnrstfile):
    """
    Analyse the cn rst file, if need, extract code-blocks from en docstring.
    """
    ...


def filter_all_files(rootdir,
                     ext='_cn.rst',
                     action=instert_codes_into_cn_rst_if_need):
    """
    find all the _en.html file, and do the action.
    """
    for root, dirs, files in os.walk(rootdir):
        for f in files:
            if f.endswith(ext):
                action(os.path.join(root, f))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='copy code-blocks from en api doc-strings.')
    parser.add_argument('--debug', dest='debug', action="store_true")
    parser.add_argument(
        'dir', type=str, help='the file directory', default='.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    filter_all_files(args.dir)
