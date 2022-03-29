# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import json
import argparse
import os.path as osp
import re

arguments = [
    # flags, dest, type, default, help
    [
        '--rst-files', 'rst_files', str, None,
        'api rst files, sperated by space'
    ],
    ['--api-info', 'api_info_file', str, None, 'api_info_all.json filename'],
]


def parse_args():
    """
    Parse input arguments
    """
    global arguments
    parser = argparse.ArgumentParser(description='check api parameters')
    parser.add_argument('--debug', dest='debug', action="store_true")
    for item in arguments:
        parser.add_argument(
            item[0], dest=item[1], help=item[4], type=item[2], default=item[3])

    args = parser.parse_args()
    return args


def check_api_parameters(rstfiles, apiinfo):
    """check function's parameters same as its origin definition.

    such as `.. py:function:: paddle.version.cuda()`
    """
    pat = re.compile(
        r'^\.\.\s+py:(method|function|class)::\s+(\S+)\s*\(\s*(.*)\s*\)\s*$')
    check_passed = []
    check_failed = []
    api_notfound = []
    for rstfile in rstfiles:
        with open(osp.join('../docs', rstfile), 'r') as rst_fobj:
            func_found = False
            for line in rst_fobj:
                mo = pat.match(line)
                if mo:
                    func_found = True
                    functype = mo.group(1)
                    if functype not in ('function', 'method'):
                        check_passed.append(rstfile)
                    funcname = mo.group(2)
                    paramstr = mo.group(3)
                    flag = False
                    for apiobj in apiinfo.values():
                        if 'all_names' in apiobj and funcname in apiobj[
                                'all_names']:
                            if 'args' in apiobj and paramstr == apiobj['args']:
                                flag = True
                            break
                    if flag:
                        check_passed.append(rstfile)
                    else:
                        check_failed.append(rstfile)
                    break
            if not func_found:
                api_notfound.append(rstfile)
    return check_passed, check_failed, api_notfound


def check_api_params_desc():
    """chech the Args Segment.

    是不是用docutils来解析rst文件的好？不要暴力正则表达式了？
    """
    ...


if __name__ == '__main__':
    args = parse_args()
    rstfiles = [fn for fn in args.rst_files.split(' ') if fn]
    apiinfo = json.load(open(args.api_info_file))
    check_passed, check_failed, api_notfound = check_api_parameters(
        rstfiles=rstfiles, apiinfo=apiinfo)
    result = True
    if check_failed:
        result = False
        print(f'check_api_parameters failed: {check_failed}')
    if api_notfound:
        print(f'check_api_parameters funcname not found in: {api_notfound}')
    if result:
        exit(0)
    else:
        exit(1)
