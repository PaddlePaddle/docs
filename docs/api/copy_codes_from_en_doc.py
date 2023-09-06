#! /bin/env python
'''
copy code-blocks from en api doc-strings.
'''
import os
import sys
import argparse
import inspect
import re
import json
import logging

api_info_dict = {}
api_name_2_id_map = {}

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


def load_api_info(api_info_json_filename):
    global api_info_dict  # update
    with open(api_info_json_filename, 'rb') as f:
        api_info_dict = json.load(f)
    for k, api_info in api_info_dict.items():
        for n in api_info.get('all_names', []):
            api_name_2_id_map[n] = k
    logger.info(
        'load %d api_infos from %s', len(api_info_dict), api_info_json_filename
    )
    logger.info('api_name_2_id_map has %d items', len(api_name_2_id_map))


def read_rst_lines_and_copy_info(cnrstfilename):
    copy_from_info = []
    rst_lines = None
    pat = re.compile(r"^(\s*)COPY-FROM\s*:\s*(.*)$", flags=re.IGNORECASE)
    with open(cnrstfilename, 'rb') as rstf:
        rst_lines = rstf.readlines()
        for lineno, line in enumerate(rst_lines):
            lineno = str(lineno)
            line = str(line)
            mo = pat.match(line)
            if mo:
                indent = len(mo.group(1))
                src = mo.group(2)
                if ':' in src:
                    src_api, cb_name = src.split(':')
                else:
                    src_api = src
                    cb_name = None
                copy_from_info.append(
                    {
                        'lineno': lineno,
                        'indent': indent,
                        'src_api': src_api.strip(),
                        'cb_name': cb_name.strip()
                        if cb_name is not None
                        else None,
                    }
                )
    return rst_lines, copy_from_info


def extract_code_blocks_from_docstr(docstr, google_style=True):
    """
    extract code-blocks from the given docstring.
    DON'T include the multiline-string definition in code-blocks.
    The *Examples* section must be the last.
    Args:
        docstr(str): docstring
        google_style(bool): if not use google_style, the code blocks will be extracted from all the parts of docstring.
    Return:
        code_blocks: A list of code-blocks, indent removed.
                     element {'name': the code-block's name, 'id': sequence id.
                              'codes': codes, 'in_examples': bool, code block in `Examples` or not,}
    """
    code_blocks = []

    mo = re.search(r"Examples?:", docstr)

    if google_style and mo is None:
        return code_blocks

    example_start = len(docstr) if mo is None else mo.start()
    docstr_describe = docstr[:example_start].splitlines()
    docstr_examples = docstr[example_start:].splitlines()

    docstr_list = []
    if google_style:
        example_lineno = 0
        docstr_list = docstr_examples
    else:
        example_lineno = len(docstr_describe)
        docstr_list = docstr_describe + docstr_examples

    lastlineindex = len(docstr_list) - 1

    cb_start_pat = re.compile(r"code-block::\s*python")
    cb_param_pat = re.compile(r"^\s*:(\w+):\s*(\S*)\s*$")

    cb_info = {}
    cb_info['cb_started'] = False
    cb_info['cb_cur'] = []
    cb_info['cb_cur_indent'] = -1
    cb_info['cb_cur_name'] = None
    cb_info['cb_cur_seq_id'] = 0

    def _cb_started():
        # nonlocal cb_started, cb_cur_name, cb_cur_seq_id
        cb_info['cb_started'] = True
        cb_info['cb_cur_seq_id'] += 1
        cb_info['cb_cur_name'] = None

    def _append_code_block(in_examples):
        # nonlocal code_blocks, cb_cur, cb_cur_name, cb_cur_seq_id
        code_blocks.append(
            {
                'codes': inspect.cleandoc("\n" + "\n".join(cb_info['cb_cur'])),
                'name': cb_info['cb_cur_name'],
                'id': cb_info['cb_cur_seq_id'],
                'in_examples': in_examples,
            }
        )

    for lineno, linecont in enumerate(docstr_list):
        if re.search(cb_start_pat, linecont):
            if not cb_info['cb_started']:
                _cb_started()
                continue
            else:
                # cur block end
                if len(cb_info['cb_cur']):
                    _append_code_block(lineno > example_lineno)
                _cb_started()  # another block started
                cb_info['cb_cur_indent'] = -1
                cb_info['cb_cur'] = []
        else:
            if cb_info['cb_started']:
                # handle the code-block directive's options
                mo_p = cb_param_pat.match(linecont)
                if mo_p:
                    if mo_p.group(1) == 'name':
                        cb_info['cb_cur_name'] = mo_p.group(2)
                    continue
                # docstring end
                if lineno == lastlineindex:
                    mo = re.search(r"\S", linecont)
                    if (
                        mo is not None
                        and cb_info['cb_cur_indent'] <= mo.start()
                    ):
                        cb_info['cb_cur'].append(linecont)
                    if len(cb_info['cb_cur']):
                        _append_code_block(lineno > example_lineno)
                    break
                # check indent for cur block start and end.
                if cb_info['cb_cur_indent'] < 0:
                    mo = re.search(r"\S", linecont)
                    if mo is None:
                        continue
                    # find the first non empty line
                    cb_info['cb_cur_indent'] = mo.start()
                    cb_info['cb_cur'].append(linecont)
                else:
                    mo = re.search(r"\S", linecont)
                    if mo is None:
                        cb_info['cb_cur'].append(linecont)
                        continue
                    if cb_info['cb_cur_indent'] <= mo.start():
                        cb_info['cb_cur'].append(linecont)
                    else:
                        if linecont[mo.start()] == '#':
                            continue
                        else:
                            # block end
                            if len(cb_info['cb_cur']):
                                _append_code_block(lineno > example_lineno)
                            cb_info['cb_started'] = False
                            cb_info['cb_cur_indent'] = -1
                            cb_info['cb_cur'] = []
    return code_blocks


def find_codeblock_needed_by_name(cb_name, codeblocks):
    for cb in codeblocks:
        if cb_name == cb.get('name', None):
            return cb
    if cb_name.isnumeric():
        cb_ind = int(cb_name)
        for cb in codeblocks:
            if cb_ind == cb.get('id', None):
                return cb
    return None


def find_codeblock_needed(cf_info):
    global api_name_2_id_map, api_info_dict  # readonly
    if cf_info['src_api'] in api_name_2_id_map:
        api_info = api_info_dict[api_name_2_id_map[cf_info['src_api']]]
        if 'docstring' in api_info:
            codeblocks = extract_code_blocks_from_docstr(
                api_info['docstring'], google_style=False
            )
            if not codeblocks:
                logger.warning('found none codeblocks for %s', str(cf_info))
                logger.warning(
                    'and the docstring is: %s', api_info['docstring']
                )
                return None

            cb_name = cf_info['cb_name']

            # we use `cb_name` first, if not exist, then use the first codeblock `in_examples` as default.
            example_codeblocks = [
                codeblock
                for codeblock in codeblocks
                if codeblock.get('in_examples')
            ]
            return (
                example_codeblocks[0]
                if cb_name is None
                else find_codeblock_needed_by_name(cb_name, codeblocks)
            )

    else:
        logger.warning('%s not in api_name_2_id_map', cf_info['src_api'])
        return None


def instert_codes_into_cn_rst_if_need(cnrstfilename):
    """
    Analyse the cn rst file, if need, extract code-blocks from en docstring.
    """
    rst_lines, copy_from_info = read_rst_lines_and_copy_info(cnrstfilename)
    update_needed = False
    pattern_doctest = re.compile(r"\s*>>>\s*#\s*doctest:\s*.*")

    if copy_from_info:
        logger.info(
            "found copy-from for %s: %s", cnrstfilename, str(copy_from_info)
        )
    for cf_info in copy_from_info:
        logger.debug('processing %s', str(cf_info))
        cb_need = find_codeblock_needed(cf_info)
        if not cb_need:
            logger.warning(
                'not found code-block for %s: %s', cnrstfilename, str(cf_info)
            )
            continue
        cb_new = []
        indent = cf_info['indent']
        cb_new.append('')  # insert a empty line in the frontend
        cb_new.append(' ' * indent + '.. code-block:: python')
        if cf_info['cb_name']:
            cb_new.append(' ' * (indent + 3) + ':name: ' + cf_info['cb_name'])
        cb_new.append('')
        indent += 4
        for line in cb_need['codes'].splitlines():
            if not pattern_doctest.match(line):
                cb_new.append(' ' * indent + line)

        rst_lines[cf_info['lineno']] = "\n".join(cb_new)
        update_needed = True
    if update_needed:
        logger.info('update ' + cnrstfilename)
        with open(cnrstfilename, 'w') as f:
            f.writelines(rst_lines)
    elif copy_from_info:
        logger.warning(
            'not found any code-blocks for %s: %s',
            cnrstfilename,
            str(copy_from_info),
        )


def filter_all_files(
    rootdir, ext='_cn.rst', action=instert_codes_into_cn_rst_if_need
):
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
        description='copy code-blocks from en api doc-strings.'
    )
    parser.add_argument('--debug', dest='debug', action="store_true")
    parser.add_argument(
        '--api-info',
        dest='api_info',
        help='the api info json file.',
        type=str,
        default='api_info_all.json',
    )
    parser.add_argument('dir', type=str, help='the file directory', default='.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    load_api_info(args.api_info)
    filter_all_files(args.dir)
