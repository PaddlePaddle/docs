import paddle
import os
import shutil
import time
import pkgutil
import types
import contextlib
import argparse

en_suffix = "_en.rst"
cn_suffix = "_cn.rst"
file_path_dict = {}


def _str2bool(s):
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gen_cn',
        type=_str2bool,
        default=False,
        help='Generate the cn documentations')
    parser.add_argument(
        '--gen_en',
        type=_str2bool,
        default=True,
        help='Generate the en documentations')
    return parser.parse_args()


def gen_doc_dir(root_path='paddle'):
    backup_path = root_path + "_" + str(int(time.time()))
    # move old dirs
    if os.path.isdir(root_path):
        os.rename(root_path, backup_path)

    os.mkdir(root_path)
    for filefiner, name, ispkg in pkgutil.walk_packages(
            path=paddle.__path__, prefix=paddle.__name__ + '.'):
        path = name.replace(".", "/")

        try:
            m = eval(name)
        except AttributeError:
            pass
        else:
            if hasattr(eval(name), "__all__"):
                os.makedirs(path)
                for api in list(set(eval(name).__all__)):
                    os.mknod(path + "/" + api + en_suffix)
                    gen = EnDocGenerator()
                    with gen.guard(path + "/" + api + en_suffix):
                        if api != 'test, get_dict':
                            gen.module_name = name
                            gen.api = api
                            gen.print_header_reminder()
                            gen.print_item()


def clean_en_files(path="./paddle"):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(en_suffix):
                os.remove(os.path.join(root, file))


def gen_en_files(root_path='paddle'):
    for filefiner, name, ispkg in pkgutil.walk_packages(
            path=paddle.__path__, prefix=paddle.__name__ + '.'):
        path = name.replace(".", "/")

        try:
            m = eval(name)
        except AttributeError:
            pass
        else:
            if hasattr(eval(name), "__all__"):
                for api in list(set(eval(name).__all__)):
                    gen = EnDocGenerator()
                    with gen.guard(path + "/" + api + en_suffix):
                        if api != 'test, get_dict':
                            gen.module_name = name
                            gen.api = api
                            gen.print_header_reminder()
                            gen.print_item()


def get_cn_files_path_dict(path=None):
    if path == None:
        path = os.getcwd() + "/../../fluid/api_cn"

    for root, dirs, files in os.walk(path):
        for file in files:
            file = file.replace(cn_suffix, "", 1)
            val = str(root + "/" + file)
            if file in file_path_dict:
                file_path_dict[file].append(val)
            else:
                file_path_dict[file] = [val]


def copy_cn_files():
    path = "./paddle"
    for root, dirs, files in os.walk(path):
        for file in files:
            f = file.replace(en_suffix, "", 1)
            if f in file_path_dict:
                key = root[len(path):] + "/" + f
                isfind = False
                for p in file_path_dict[f]:
                    if p.find(key) != -1:
                        src = p + cn_suffix
                        dst = root + "/" + f + cn_suffix
                        shutil.copy(src, dst)
                        isfind = True
                        break
                if not isfind:
                    src = find_max_size_file(file_path_dict[f]) + cn_suffix
                    dst = root + "/" + f + cn_suffix
                    shutil.copy(src, dst)


def check_cn_en_match(path="./paddle", diff_file="en_cn_files_diff"):
    fo = open(diff_file, 'w')
    fo.write("exist\tnot_exits\n")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(en_suffix):
                cf = file.replace(en_suffix, cn_suffix)
                if not os.path.exists(root + "/" + cf):
                    fo.write(
                        os.path.join(root, file) + "\t" + os.path.join(
                            root, cf) + "\n")

            elif file.endswith(cn_suffix):
                ef = file.replace(cn_suffix, en_suffix)
                if not os.path.exists(root + "/" + ef):
                    fo.write(
                        os.path.join(root, file) + "\t" + os.path.join(
                            root, ef) + "\n")
    fo.close()


def find_max_size_file(files):
    max_size = 0
    file = ""
    for f in files:
        if os.path.getsize(f + cn_suffix) > max_size:
            max_size = os.path.getsize(f + cn_suffix)
            file = f

    return file


class EnDocGenerator(object):
    def __init__(self, name=None, api=None):
        self.module_name = name
        self.api = api
        self.stream = None

    @contextlib.contextmanager
    def guard(self, filename):
        assert self.stream is None, "stream must be None"
        self.stream = open(filename, 'w')
        yield
        self.stream.close()
        self.stream = None

    def print_item(self):
        try:
            m = eval(self.module_name + "." + self.api)
        except AttributeError:
            #print("attribute error: module_name=" + self.module_name  + ", api=" + self.api)
            pass
        else:
            if isinstance(
                    eval(self.module_name + "." + self.api), types.TypeType):
                self.print_class()
            elif isinstance(
                    eval(self.module_name + "." + self.api),
                    types.FunctionType):
                self.print_function()

    def print_header_reminder(self):
        self.stream.write('''..  THIS FILE IS GENERATED BY `gen_doc.{py|sh}`
    !DO NOT EDIT THIS FILE MANUALLY!

''')

    def _print_ref_(self):
        self.stream.write(".. _api_{0}_{1}:\n\n".format("_".join(
            self.module_name.split(".")), self.api))

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

    def print_class(self):
        self._print_ref_()
        self._print_header_(self.api, dot='-', is_title=False)
        if "fluid.dygraph" in self.module_name:
            self.stream.write('''..  autoclass:: paddle.{0}.{1}
    :members:
    :noindex:

'''.format(self.module_name, self.api))
        elif "fluid.optimizer" in self.module_name:
            self.stream.write('''..  autoclass:: paddle.{0}.{1}
    :members:
    :inherited-members:
    :exclude-members: apply_gradients, apply_optimize, backward, load
    :noindex:

'''.format(self.module_name, self.api))
        else:
            self.stream.write('''..  autoclass:: paddle.{0}.{1}
    :members:
    :inherited-members:
    :noindex:

'''.format(self.module_name, self.api))

    def print_function(self):
        self._print_ref_()
        self._print_header_(self.api, dot='-', is_title=False)
        self.stream.write('''..  autofunction:: paddle.{0}.{1}
    :noindex:

'''.format(self.module_name, self.api))


if __name__ == "__main__":
    args = parse_args()
    if args.gen_cn:
        gen_doc_dir()
        get_cn_files_path_dict()
        copy_cn_files()

    clean_en_files()
    if args.gen_en:
        gen_en_files()
        check_cn_en_match()
