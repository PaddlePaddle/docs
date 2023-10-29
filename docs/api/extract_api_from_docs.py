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
from contextlib import contextmanager
import docutils
import docutils.core
import docutils.nodes
import docutils.parsers.rst
import markdown

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
                    if mo is None:
                        continue
                    cb_cur_indent = mo.start()
                    cb_cur.append(linecont)
                else:
                    mo = re.search(r"\S", linecont)
                    if mo is None:
                        continue
                    if cb_cur_indent <= mo.start():
                        cb_cur.append(linecont)
                    else:
                        if linecont[mo.start()] == '#':
                            continue
                        else:
                            # block end
                            if len(cb_cur):
                                code_blocks.append(
                                    inspect.cleandoc("\n".join(cb_cur))
                                )
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
    for cbit in pat.finditer(docstr):
        code_blocks.append(inspect.cleandoc(cbit.group()))
    # logger.info('extracted %d code blocks.', len(code_blocks))
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
        if in_docstr:
            continue
        sharp_ind = line.find('#')
        mo_i = import_pat.search(line)
        if mo_i:
            if (sharp_ind < 0 or mo_i.start() < sharp_ind) and mo_i.group(
                1
            ).startswith('paddle'):
                api_set.add(f'{mo_i.group(1)}.{mo_i.group(2)}')
        else:
            mo_n = normal_pat.finditer(line)
            for mo in mo_n:
                if sharp_ind < 0 or mo.start() < sharp_ind:
                    api_set.add(mo.group(1))
    return api_set


def extract_api_from_file(filename):
    api_set = set()
    codeblocks = extract_code_blocks_from_file(filename)
    logger.info('find %d code-blocks from %s.', len(codeblocks), filename)
    for cb in codeblocks:
        api_set.update(find_all_paddle_api_from_code_block(cb))
    logger.info('find %d apis from %s.', len(api_set), filename)
    return api_set


def extract_doc_title_from_file(filename):
    r = os.path.splitext(filename)
    if len(r) != 2:
        return None
    if r[1].lower() == '.md':
        return extract_md_title(filename)
    elif r[1].lower() == '.rst':
        return extract_rst_title(filename)
    return None


@contextmanager
def find_node_by_class(doctree, node_class, remove):
    """Find the first node of the specified class."""
    index = doctree.first_child_matching_class(node_class)
    if index is not None:
        yield doctree[index]
        if remove:
            del doctree[index]
    else:
        yield


def extract_rst_title(filename):
    overrides = {
        # Disable the promotion of a lone top-level section title to document
        # title (and subsequent section title to document subtitle promotion).
        'docinfo_xform': 0,
        'initial_header_level': 2,
    }
    with open(filename, 'r') as fileobj:
        doctree = docutils.core.publish_doctree(
            fileobj.read(), settings_overrides=overrides
        )
        with find_node_by_class(
            doctree, docutils.nodes.title, remove=True
        ) as node:
            if node is not None:
                return node.astext()
    return None


def extract_params_desc_from_rst_file(filename, section_title='参数'):
    overrides = {
        # Disable the promotion of a lone top-level section title to document
        # title (and subsequent section title to document subtitle promotion).
        'docinfo_xform': 0,
        'initial_header_level': 2,
    }
    with open(filename, 'r') as fileobj:
        doctree = docutils.core.publish_doctree(
            fileobj.read(), settings_overrides=overrides
        )
        found = False
        for child in doctree.children:
            if isinstance(child, docutils.nodes.section) and isinstance(
                child.children[0], docutils.nodes.title
            ):
                sectitle = child.children[0].astext()
                if isinstance(section_title, (list, tuple)):
                    for st in section_title:
                        if sectitle.startswith(st):
                            found = True
                            break
                else:
                    if sectitle.startswith(section_title):
                        found = True
                if found:
                    return child
    return None


def extract_md_title(filename):
    with open(filename, 'r') as fileobj:
        html = markdown.markdown(fileobj.read())
        mo = re.search(r'<h1>(.*?)</h1>', html)
        if mo:
            mos = re.search(r'<strong>(.*?)</strong>', mo.group(1))
            return mos.group(1) if mos else mo.group(1)
    return None


def format_filename(filename):
    """
    Format the filename

    filename may be "/FluidDoc/docs/guides/xxx" or "../guides/xxx", format it as "guides/xxx".
    function get_all_files does not format it.
    """
    rp = os.path.realpath(filename)
    pat_str = 'docs/'  # if the structure changed, update this pattern
    ind = rp.rindex(pat_str)
    if ind >= 0:
        return rp[ind + len(pat_str) :]
    return filename


def extract_all_infos(docdirs):
    apis_dict = {}
    file_titles = {}
    for p in docdirs:
        filelist = get_all_files(p)
        for fn in filelist:
            ffn = format_filename(fn)
            file_titles[ffn] = extract_doc_title_from_file(fn)
            apis = extract_api_from_file(fn)
            if len(apis):
                apis_dict[ffn] = list(apis)
    return apis_dict, file_titles


def ref_role(role, rawtext, text, lineno, inliner, options=None, content=None):
    '''dummy ref role'''
    ref_target = text
    node = docutils.nodes.reference(rawtext, text)
    return [node], []


docutils.parsers.rst.roles.register_canonical_role('ref', ref_role)
docutils.parsers.rst.roles.register_local_role('ref', ref_role)


class PyFunctionDirective(docutils.parsers.rst.Directive):
    '''dummy py:function directive

    see https://docutils-zh-cn.readthedocs.io/zh_CN/latest/howto/rst-roles.html
    '''

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    has_content = True

    def run(self):
        text = '\n'.join(self.content)
        thenode = docutils.nodes.title(text, text)
        return [thenode]


docutils.parsers.rst.directives.register_directive(
    'py:function', PyFunctionDirective
)  # as abs_cn.rst
docutils.parsers.rst.directives.register_directive(
    'py:class', PyFunctionDirective
)  # as Tensor_cn.rst
docutils.parsers.rst.directives.register_directive(
    'py:method', PyFunctionDirective
)  # as grad_cn.rst


class ToctreeDirective(docutils.parsers.rst.Directive):
    '''dummy toctree directive'''

    required_arguments = 1
    optional_arguments = 5
    has_content = True

    def run(self):
        text = self.arguments[0]
        thenode = None
        return []


docutils.parsers.rst.directives.register_directive('toctree', ToctreeDirective)

arguments = [
    # flags, dest, type, default, help
    [
        '--output',
        'output',
        str,
        'called_apis_from_docs.json',
        'output filename. default: called_apis_from_docs.json',
    ],
]


def parse_args():
    """
    Parse input arguments
    """
    global arguments
    parser = argparse.ArgumentParser(
        description='extract all the called apis from md or reST files.'
    )
    parser.add_argument(
        'dir',
        type=str,
        help='travel all the files include this directory',
        default='.',
        nargs='+',
    )
    for item in arguments:
        parser.add_argument(
            item[0], dest=item[1], help=item[4], type=item[2], default=item[3]
        )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    logger.setLevel(logging.DEBUG)
    apis_dict, file_titles = extract_all_infos(args.dir)
    import json

    with open(args.output, 'w') as f:
        json.dump(apis_dict, f, indent=4)
    r = os.path.splitext(args.output)
    with open(f'{r[0]}-titles{r[1]}', 'w') as f:
        json.dump(file_titles, f, indent=4)

    print('Done')
