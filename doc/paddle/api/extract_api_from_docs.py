#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
Extract used apis from markdown and reStructured documents.
"""

import re
import inspect
import os
import argparse
import logging

logger = logging.getLogger()
if logger.handlers:
    # we assume the first handler is the one we want to configure
    console = logger.handlers[0]
else:
    console = logging.StreamHandler()
    logger.addHandler(console)
console.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"))


def extract_code_blocks_from_rst(docstr):
    """
    extract code-blocks from the given docstring.
    DON'T include the multiline-string definition in code-blocks.
    Args:
        docstr - docstring
    Return:
        A list of code-blocks, indent removed.
    """
    code_blocks = []
    ds_list = docstr.expandtabs(tabsize=4).split("\n")
    lastlineindex = len(ds_list) - 1
    cb_started = False
    cb_start_pat = re.compile(r"((code)|(code-block))::\s*i?python[23]?")
    cb_cur = []
    cb_cur_indent = -1
    for lineno, linecont in enumerate(ds_list):
        if re.search(cb_start_pat, linecont):
            if not cb_started:
                cb_started = True
                continue
            else:
                # cur block end
                if len(cb_cur):
                    code_blocks.append(inspect.cleandoc("\n".join(cb_cur)))
                cb_started = True  # another block started
                cb_cur_indent = -1
                cb_cur = []
        else:
            # check indent for cur block ends.
            if cb_started:
                if lineno == lastlineindex:
                    mo = re.search(r"\S", linecont)
                    if mo is not None and cb_cur_indent <= mo.start():
                        cb_cur.append(linecont)
                    if len(cb_cur):
                        code_blocks.append(inspect.cleandoc("\n".join(cb_cur)))
                    break
                if cb_cur_indent < 0:
                    mo = re.search(r"\S", linecont)
                    if mo is None: continue
                    cb_cur_indent = mo.start()
                    cb_cur.append(linecont)
                else:
                    mo = re.search(r"\S", linecont)
                    if mo is None: continue
                    if cb_cur_indent <= mo.start():
                        cb_cur.append(linecont)
                    else:
                        if linecont[mo.start()] == '#':
                            continue
                        else:
                            # block end
                            if len(cb_cur):
                                code_blocks.append(
                                    inspect.cleandoc("\n".join(cb_cur)))
                            cb_started = False
                            cb_cur_indent = -1
                            cb_cur = []
    logger.info('extracted %d code blocks.', len(code_blocks))
    return code_blocks


def extract_code_blocks_from_md(docstr):
    """
    Extract code blocks from markdown content.
    """
    code_blocks = []
    pat = re.compile(r"```i?python[23]?(.*?)```", re.MULTILINE + re.DOTALL)
    res = pat.search(docstr)
    if res:
        for cb in res.groups():
            code_blocks.append(inspect.cleandoc(cb))
    logger.info('extracted %d code blocks.', len(code_blocks))
    return code_blocks


def extract_code_blocks_from_file(filename):
    r = os.path.splitext(filename)
    ext = r[1].lower()
    if ext == '.md':
        return extract_code_blocks_from_md(open(filename, 'r').read())
    elif ext == '.rst':
        return extract_code_blocks_from_rst(open(filename, 'r').read())
    else:
        return []


def get_all_files(p):
    """
    Get a filename list from the dir.
    """
    filelist = []
    for path, dirs, files in os.walk(p):
        for filename in files:
            r = os.path.splitext(filename)
            logger.info('%s found', filename)
            if len(r) == 2 and r[1].lower() in ['.md', '.rst']:
                filelist.append(os.path.join(path, filename))
    logger.info('find %d files from %s.', len(filelist), p)
    return filelist


def find_all_paddle_api_from_code_block(cbstr):
    """
    Find All Paddle Api
    """
    # case 1.1: import paddle.abs  # ignore
    # case 1.2: from paddle.vision.transforms import ToTensor
    # case 1.3: import paddle.vision.transforms.ToTensor as ToTensor  # ignore
    # case 2: train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
    # case 3: in comments
    # case 4: in docstring
    api_set = set()
    ds_list = cbstr.split("\n")
    import_pat = re.compile(r'from\s+([\.\w]+)\s+import\s+(\w+)')
    normal_pat = re.compile(r'(paddle\.[\.\w]+)')
    docstr_pat = re.compile(r'((\'{3})|(\"{3}))')
    in_docstr = False
    for line in ds_list:
        line = line.strip()
        for mo in docstr_pat.finditer(line):
            in_docstr = not in_docstr
        if in_docstr: continue
        sharp_ind = line.find('#')
        mo_i = import_pat.search(line)
        if mo_i:
            if (sharp_ind < 0 or mo_i.start() < sharp_ind
                ) and mo_i.group(1).startswith('paddle'):
                api_set.add('{}.{}'.format(mo_i.group(1), mo_i.group(2)))
        else:
            mo_n = normal_pat.finditer(line)
            for mo in mo_n:
                if sharp_ind < 0 or mo.start() < sharp_ind:
                    api_set.add(mo.group(1))
    return api_set


def extract_api_from_file(filename):
    api_set = set()
    codeblocks = extract_code_blocks_from_file(filename)
    for cb in codeblocks:
        api_set.update(find_all_paddle_api_from_code_block(cb))
    logger.info('find %d apis from %s.', len(api_set), filename)
    return api_set


arguments = [
    # flags, dest, type, default, help
    [
        '--output', 'output', str, 'called_apis_from_docs.json',
        'output filename. default: called_apis_from_docs.json'
    ],
]


def parse_args():
    """
    Parse input arguments
    """
    global arguments
    parser = argparse.ArgumentParser(
        description='extract all the called apis from md or reST files.')
    parser.add_argument(
        'dir',
        type=str,
        help='travel all the files include this directory',
        default='.')
    for item in arguments:
        parser.add_argument(
            item[0], dest=item[1], help=item[4], type=item[2], default=item[3])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print('{}'.format(args))
    logger.setLevel(logging.DEBUG)
    filelist = get_all_files(args.dir)
    apis_dict = {}
    for fn in filelist:
        apis = extract_api_from_file(fn)
        if len(apis):
            apis_dict[fn] = list(apis)
    with open(args.output, 'w') as f:
        import json
        json.dump(apis_dict, f, indent=4)

    print('Done')
