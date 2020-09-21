#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import sys
import types
import os
import contextlib
import paddle.fluid as fluid
import paddle.tensor as tensor
import paddle.nn as nn
import paddle.optimizer as optimizer

#import paddle.complex as complex
#import paddle.framework as framework


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submodules', nargs="*")
    parser.add_argument(
        '--module_name',
        type=str,
        help='Generate the documentation of which module')
    parser.add_argument(
        '--module_prefix', type=str, help='Generate the prefix of module')
    parser.add_argument(
        '--output',
        type=str,
        help='Output file or output directory for output rst')
    parser.add_argument(
        '--output_name',
        type=str,
        help='Output file or output directory for output rst')
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output file or output directory for output rst')
    parser.add_argument(
        '--to_multiple_files',
        type=bool,
        default=False,
        help='Whether to separate to multiple files')

    return parser.parse_args()

    def print_item(self, name):
        item = getattr(self.module, name, None)
        if item is None:
            return
        if isinstance(item, types.TypeType):
            self.print_class(name)
        elif isinstance(item, types.FunctionType):
            self.print_method(name)
        else:
            pass


class DocGenerator(object):
    def __init__(self, module_name=None, module_prefix=None):
        self.module_name = module_name
        self.module_prefix = module_prefix
        self.stream = None

    @contextlib.contextmanager
    def guard(self, filename):
        assert self.stream is None, "stream must be None"
        self.stream = open(filename, 'w')
        yield
        self.stream.close()
        self.stream = None

    def print_submodule(self, submodule_name):
        submodule = getattr(self.module, submodule_name)
        if submodule is None:
            raise ValueError(
                "Cannot find submodule {0}".format(submodule_name))
        self.print_section(submodule_name)

        for item in sorted(submodule.__all__, key=str.lower):
            self.print_item(item)

    def print_current_module(self):
        for item in sorted(self.module.__all__, key=str.lower):
            self.print_item(item)

    def print_section(self, name):
        self._print_header_(name, dot='=', is_title=False)

    def print_item(self, name, output_name):
        item = getattr(self.module, name, None)
        if isinstance(item, types.TypeType):
            self.print_class(name)
        elif isinstance(item, types.FunctionType):
            self.print_method(name)
        else:
            self.stream.close()
            path = os.getcwd() + "/" + output_name + "/" + name + ".rst"
            if name != "PipeReader":
                os.remove(path)

    def print_class(self, name):
        self._print_ref_(name)
        self._print_header_(name, dot='-', is_title=False)
        if "fluid.dygraph" in self.module_prefix:
            self.stream.write('''..  autoclass:: paddle.{0}.{1}
    :members:
    :noindex:

'''.format(self.module_prefix, name))
        elif "fluid.optimizer" in self.module_prefix:
            self.stream.write('''..  autoclass:: paddle.{0}.{1}
    :members:
    :inherited-members:
    :exclude-members: apply_gradients, apply_optimize, backward, load
    :noindex:

'''.format(self.module_prefix, name))
        else:
            self.stream.write('''..  autoclass:: paddle.{0}.{1}
    :members:
    :inherited-members:
    :noindex:

'''.format(self.module_prefix, name))

    def print_method(self, name):
        self._print_ref_(name)
        self._print_header_(name, dot='-', is_title=False)
        self.stream.write('''..  autofunction:: paddle.{0}.{1}
    :noindex:

'''.format(self.module_prefix, name))

    def print_header_reminder(self):
        self.stream.write('''..  THIS FILE IS GENERATED BY `gen_doc.{py|sh}`
    !DO NOT EDIT THIS FILE MANUALLY!

''')

    def _print_header_(self, name, dot, is_title):
        dot_line = dot * len(name)
        if is_title:
            self.stream.write(dot_line)
            self.stream.write('\n')
        self.stream.write(name)
        self.stream.write('\n')
        self.stream.write(dot_line)
        self.stream.write('\n')
        self.stream.write('\n')

    def _print_ref_(self, name):
        self.stream.write(".. _api_{0}_{1}:\n\n".format("_".join(
            self.module_prefix.split(".")), name))


def generate_doc(module_name, module_prefix, output, output_name,
                 to_multiple_files, output_dir):
    if module_name == "":
        module_name = None

    if module_prefix == "":
        module_prefix = None

    gen = DocGenerator()

    if module_name is None:
        gen.module = eval(output_name)
        gen.module_name = str(output_name)
    else:
        gen.module = eval(output_name)
        for each_module_name in module_name.split('.'):
            if not hasattr(gen.module, each_module_name):
                raise ValueError("Cannot find fluid.{0}".format(module_name))
            else:
                gen.module = getattr(gen.module, each_module_name)

        gen.module_name = output_name + "." + module_name

    if module_prefix is None:
        gen.module_prefix = gen.module_name
    else:
        gen.module_prefix = output_name + "." + module_prefix

    dirname = output if to_multiple_files else os.path.dirname(output)

    if output_dir != None:
        dirname = output_dir + "/" + dirname
        output = output_dir + "/" + output

    if len(dirname) > 0 and (not os.path.exists(dirname) or
                             not os.path.isdir(dirname)):
        os.makedirs(dirname)

    if not to_multiple_files:
        header_name = gen.module_name
        if module_prefix is not None:
            prefix_len = len(gen.module_prefix)
            assert gen.module_prefix == gen.module_name[0:prefix_len],    \
                "module_prefix must be prefix of module_name"
            diff_name = gen.module_name[prefix_len + 1:]
            if diff_name != "":
                header_name = diff_name
    else:
        header_name = None

    if not to_multiple_files:
        with gen.guard(output):
            gen.print_header_reminder()
            gen._print_header_(header_name, dot='=', is_title=True)
            gen.print_current_module()
    else:
        apis = sorted(gen.module.__all__, key=str.lower)
        for api in apis:
            header_name = api
            with gen.guard(os.path.join(output, api + '.rst')):
                gen.print_header_reminder()
                gen.print_item(api, output_name)


def main():
    args = parse_arg()
    generate_doc(args.module_name, args.module_prefix, args.output,
                 args.output_name, args.to_multiple_files, args.output_dir)


if __name__ == '__main__':
    main()
