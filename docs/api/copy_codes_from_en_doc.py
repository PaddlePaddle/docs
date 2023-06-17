#! /bin/env python
'''
copy code-blocks from en api doc-strings.
'''
import os
import sys
import argparse
import re
import json
import logging
from gen_doc import extract_code_blocks_from_docstr

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
    with open(api_info_json_filename, 'r') as f:
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
    with open(cnrstfilename, 'r') as rstf:
        rst_lines = rstf.readlines()
        for lineno, line in enumerate(rst_lines):
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

            # we use `cb_name` first, if not exist, then use the first codeblock without name, or codeblocks[0].
            cb = None
            if cb_name is not None:
                cb = find_codeblock_needed_by_name(cb_name, codeblocks)
            else:
                for _cb in codeblocks:
                    if _cb.get('name', None) is None:
                        cb = _cb
                        break

            return cb if cb is not None else codeblocks[0]
    else:
        logger.warning('%s not in api_name_2_id_map', cf_info['src_api'])
        return None


def instert_codes_into_cn_rst_if_need(cnrstfilename):
    """
    Analyse the cn rst file, if need, extract code-blocks from en docstring.
    """
    rst_lines, copy_from_info = read_rst_lines_and_copy_info(cnrstfilename)
    update_needed = False
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
