import json
import argparse
import sys

source_to_doc_dict = {}
SYSTEM_MESSAGE_WARNING = 'System Message: WARNING'
SYSTEM_MESSAGE_ERROR = 'System Message: ERROR'
EN_HTML_EXTENSION = '_en.html'

arguments = [
    [
        '--py_files',
        'py_files',
        str,
        None,
        'api python files, sperated by space',
    ],
    [
        '--api_info_file',
        'api_info_file',
        str,
        None,
        'api_info_all.json filename',
    ],
    ['--output_path', 'output_path', str, None, 'output_path'],
]


def parse_args():
    """
    Parse input arguments
    """
    global arguments
    parser = argparse.ArgumentParser(
        description='system message check parameters'
    )
    parser.add_argument('--debug', dest='debug', action="store_true")
    for item in arguments:
        parser.add_argument(
            item[0], dest=item[1], help=item[4], type=item[2], default=item[3]
        )

    args = parser.parse_args()
    return args


def build_source_file_to_doc_file_dict(api_info):
    """
    构建API源码文件-文档文件列表的字典
    一个源码文件可能对应多个文档文件
    parameter: api_info 由 api_info_all.json 解析出的json对象
    return: dict{scr_file: [doc_filenames]}
    """
    for k, v in api_info.items():
        if 'src_file' in v and 'doc_filename' in v:
            src_file = v['src_file']
            doc_file = v['doc_filename']
            if src_file not in source_to_doc_dict:
                source_to_doc_dict[src_file] = []
            source_to_doc_dict[src_file].append(doc_file)
    return source_to_doc_dict


def check_system_message_in_doc(doc_file):
    """
    检查英文文档中是否出现 System Message: Warning/Error 的字符串
    parameter: doc_file 英文文档的html文件
    return: True or False
    """
    pass_check = True
    with open(doc_file, 'r') as f:
        for line, row in enumerate(f):
            if SYSTEM_MESSAGE_WARNING in row:
                print(
                    'ERROR: ',
                    doc_file,
                    ' line: ',
                    line,
                    'has ',
                    SYSTEM_MESSAGE_WARNING,
                )
                pass_check = False
            if SYSTEM_MESSAGE_ERROR in row:
                print(
                    'ERROR: ',
                    doc_file,
                    'line: ',
                    line,
                    'has ',
                    SYSTEM_MESSAGE_ERROR,
                )
                pass_check = False
    return pass_check


if __name__ == '__main__':
    args = parse_args()
    py_files = list(args.py_files.split('\n'))
    # 此处获取的全路径是 python/paddle/amp/auto_cast.py，在 api_info_all.json 中的路径是 /paddle/amp/auto_cast.py
    # 做字符串替换
    for i in range(len(py_files)):
        if py_files[i].startswith('python/'):
            py_files[i] = py_files[i][6:]
    api_info = json.load(open(args.api_info_file))
    output_path = args.output_path
    build_source_file_to_doc_file_dict(api_info)
    error_files = set()
    for i in py_files:
        if i not in source_to_doc_dict:
            continue
        doc_files = source_to_doc_dict[i]
        print(i, ' has doc file: ', doc_files)
        # check 'System Message: WARNING/ERROR' in api doc file
        for doc_file in doc_files:
            check = check_system_message_in_doc(
                output_path + doc_file + EN_HTML_EXTENSION
            )
            if not check:
                error_files.add(i + ' - ' + doc_file + EN_HTML_EXTENSION)
    if error_files:
        print('error files: ', error_files)
        print(
            'ERROR: these docs exsits System Message: WARNING/ERROR, please check and fix them'
        )
        sys.exit(1)
