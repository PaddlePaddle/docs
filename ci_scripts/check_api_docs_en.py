import json
import argparse
import sys

source_to_doc_dict = {}
SYSTEM_MESSAGE_WARNING = 'System Message: WARNING'
SYSTEM_MESSAGE_ERROR = 'System Message: ERROR'


def build_source_file_to_doc_file_dict(api_info):
    
    for k, v in api_info.items():
        if 'src_file' in v and 'doc_filename' in v:
            src_file = v['src_file']
            doc_file = v['doc_filename']
            if src_file not in source_to_doc_dict:
                source_to_doc_dict[src_file] = []
            source_to_doc_dict[src_file].append(doc_file)
    return source_to_doc_dict


def check_system_message_in_doc(doc_file):
    pass_check = True
    with open(doc_file, 'r') as f:
        for line, i in enumerate(f):
            if SYSTEM_MESSAGE_WARNING in line:
                print('ERROR: ', doc_file, ' line: ', line, 'has ', SYSTEM_MESSAGE_WARNING)
                pass_check = False
            if 'System Message: ERROR' in line:
                print('ERROR: ', doc_file,  'line: ', line, 'has ', SYSTEM_MESSAGE_WARNING)
                pass_check = False
    return pass_check


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='system message check parameters')
    args = parser.parse_args()
    py_files = [fn for fn in args.py_files.split(' ') if fn]
    api_info = json.load(open(args.api_info_file))
    output_path = args.output_path
    build_source_file_to_doc_file_dict(api_info)
    error_files = set()
    for i in py_files:
        if i not in source_to_doc_dict:
            print(i, ' has no doc file')
            continue
        doc_file = source_to_doc_dict[i]
        # check 'System Message: WARNING/ERROR' in api doc file
        check = check_system_message_in_doc(output_path + doc_file)
        if not check:
            error_files.add(doc_file)
    if error_files:
        sys.exit(1)
