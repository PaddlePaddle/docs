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
same_api_map = {}
alias_api_map = {}
not_display_doc_map = {}
display_doc_map = {}
api_set = set()


def get_all_api(root_path='paddle'):
    for filefiner, name, ispkg in pkgutil.walk_packages(
            path=paddle.__path__, prefix=paddle.__name__ + '.'):
        try:
            m = eval(name)
        except AttributeError:
            pass
        else:
            if hasattr(eval(name), "__all__"):
                #may have duplication of api
                for api in list(set(eval(name).__all__)):
                    api_all = name + "." + api
                    if "," in api:
                        continue

                    try:
                        fc_id = id(eval(api_all))
                    except AttributeError:
                        pass
                    else:
                        api_set.add(api_all)


def get_all_same_api():
    for api in api_set:
        fc_id = id(eval(api))
        if fc_id in same_api_map:
            same_api_map[fc_id].append(api)
        else:
            same_api_map[fc_id] = [api]


def get_not_display_doc_list(file="./not_display_doc_list"):
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            not_display_doc_map[line] = 1


def get_display_doc_map(file="./display_doc_list"):
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            display_doc_map[line] = 1


def get_alias_mapping(file="./alias_api_mapping"):
    with open(file, 'r') as f:
        for line in f.readlines():
            t = line.strip().split('\t')
            real_api = t[0].strip()
            alias_api = t[1].strip()
            alias_api_map[real_api] = alias_api


def is_filter_api(api):
    #if api in display_list, just return False
    if api in display_doc_map:
        return False

    #check in api in not_display_list
    for key in not_display_doc_map:
        #find the api
        if key == api:
            return True
        #find the module
        if api.startswith(key):
            k_segs = key.split(".")
            a_segs = api.split(".")
            if k_segs[len(k_segs) - 1] == a_segs[len(k_segs) - 1]:
                return True

    #check api in alias map
    if alias_api_map.has_key(api):
        return False

    #check api start with paddle.fluid
    #if has no alias, return True
    #if has alias also in paddle.fluid, return True
    #if has alias in other module, return False
    same_apis = same_api_map[id(eval(api))]
    if api.startswith("paddle.fluid"):
        all_fluid_flag = True
        for x in same_apis:
            if not x.startswith("paddle.fluid"):
                all_fluid_flag = False

        if all_fluid_flag:
            return True

    #if the api in alias_map key, others api is alias api
    for x in same_apis:
        if alias_api_map.has_key(x):
            return True

    if len(same_apis) > 1:
        # find shortest path of api as the real api
        # others api as the alias api
        shortest = len(same_apis[0].split("."))
        for x in same_apis:
            if len(x.split(".")) < shortest:
                shortest = len(x.split("."))

        if len(api.split(".")) == shortest:
            return False
        else:
            return True
    return False


def gen_en_files(root_path='paddle'):
    backup_path = root_path + "_" + str(int(time.time()))

    for api in api_set:
        if is_filter_api(api):
            continue

        doc_file = api.split(".")[-1]
        path = "/".join(api.split(".")[0:-1])
        if not os.path.exists(path):
            os.makedirs(path)
        f = api.replace(".", "/")
        os.mknod(f + en_suffix)
        gen = EnDocGenerator()
        with gen.guard(f + en_suffix):
            gen.module_name = ".".join(api.split(".")[0:-1])
            gen.api = doc_file
            gen.print_header_reminder()
            gen.print_item()


def clean_en_files(path="./paddle"):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(en_suffix):
                os.remove(os.path.join(root, file))


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
        self.stream.write('''..  autofunction:: {0}.{1}
    :noindex:

'''.format(self.module_name, self.api))


if __name__ == "__main__":
    get_all_api()
    get_not_display_doc_list()
    get_display_doc_map()
    get_all_same_api()
    get_alias_mapping()

    clean_en_files()
    gen_en_files()
    check_cn_en_match()
