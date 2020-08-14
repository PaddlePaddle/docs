import argparse
import sys
import types
import os
import contextlib


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api_path',
        type=str,
        default='paddle.nn.functional.l1_loss',
        help='the function/class path')
    parser.add_argument(
        '--is_class',
        type=str,
        default='False',
        help='whether function or class, False means function')
    return parser.parse_args()


def add_index(en_doc_review_dir, api_name):

    stream = open(en_doc_review_dir + '.rst', 'a')
    stream.write('    review_tmp/' + api_name + '.rst\n')
    stream.close()
    print('add index to ' + en_doc_review_dir + '.rst success')


def add_file(en_doc_review_dir, api_path, is_class=False):

    api_path_list = api_path.split('.')
    api_name = api_path_list[-1]
    api_title = '_'.join(api_path_list[1:])

    stream = open(en_doc_review_dir + '/' + api_name + '.rst', 'w')
    stream.write('.. _api_' + api_title + ':\n')
    stream.write('\n')
    stream.write(api_name + '\n')
    for i in range(max(9, len(api_name))):
        stream.write('-')
    stream.write('\n')
    stream.write('\n')

    if is_class == 'True':
        stream.write('..  autoclass:: ' + api_path + '\n')
        stream.write('    :members:\n')
        stream.write('    :inherited-members:\n')
    else:
        stream.write('..  autofunction:: ' + api_path + '\n')

    stream.write('    :noindex:\n')
    stream.close()
    print('add' + en_doc_review_dir + '/' + api_name + '.rst success')


def main():
    args = parse_arg()
    api_path = args.api_path
    is_class = args.is_class
    api_name = api_path.split('.')[-1]

    fluid_doc_path = os.getcwd()
    en_doc_review_dir = fluid_doc_path + '/doc/fluid/api/review_tmp'

    add_index(en_doc_review_dir, api_name)
    add_file(en_doc_review_dir, api_path, is_class)


if __name__ == '__main__':
    main()
