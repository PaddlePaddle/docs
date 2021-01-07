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
id_real_api_map = {}


def get_all_api(root_path='paddle'):
    for filefinder, name, ispkg in pkgutil.walk_packages(
            path=paddle.__path__, prefix=paddle.__name__ + '.'):
        # not show paddle.reader APIs
        if name.startswith("paddle.reader"):
            continue
        try:
            m = eval(name)
        except AttributeError:
            pass
        else:
            if hasattr(eval(name), "__all__"):
                #may have duplication of api
                for api in set(eval(name).__all__):
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
            if "\t" in line:
                t = line.strip().split('\t')
            else:
                t = line.strip().split('    ')
            if len(t) < 2:
                print('line "', line, '" splited to ', t)
                continue
            real_api = t[0].strip()
            alias_apis = t[1].strip().split(',')
            alias_api_map[real_api] = alias_apis

            try:
                m = eval(real_api)
            except AttributeError:
                print("AttributeError:", real_api)
                pass
            else:
                id_real_api_map[id(eval(real_api))] = real_api


def filter_same_api():
    # filter same_apis by display list and not display list
    for k in list(same_api_map.keys()):
        same_apis = same_api_map[k]
        if is_display_apis(same_apis):
            continue

        if is_not_display_apis(same_apis):
            del same_api_map[k]


def find_real_api(api_list):
    # find the real_api
    if len(api_list) == 1:
        return api_list[0]
    else:
        return choose_real_api(api_list)


def choose_real_api(api_list):
    api = api_list[0]
    # the api in id_real_api_map, return real_api
    if id(eval(api)) in id_real_api_map:
        return id_real_api_map[id(eval(api))]
    else:
        # in case of alias_mapping not contain the apis
        # try to find shortest path of api as the real api
        shortest = len(api_list[0].split("."))
        shorttest_api = api_list[0]
        for x in api_list:
            if len(x.split(".")) < shortest:
                shortest = len(x.split("."))
                shorttest_api = x

        return shorttest_api


# check if any of the same apis in display list
def is_display_apis(api_list):
    for api in api_list:
        if api in display_doc_map:
            return True
    return False


def is_not_display_apis(api_list):
    for api in api_list:
        #check api in not_display_list
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
    return False


def gen_en_files(root_path='paddle', api_label_file="api_label"):
    api_f = open(api_label_file, 'w')

    for api_list in same_api_map.values():
        real_api = find_real_api(api_list)

        module_name = ".".join(real_api.split(".")[0:-1])
        doc_file = real_api.split(".")[-1]

        if isinstance(eval(real_api), types.ModuleType):
            continue

        path = "/".join(real_api.split(".")[0:-1])
        if not os.path.exists(path):
            os.makedirs(path)
        f = real_api.replace(".", "/") + en_suffix
        if os.path.exists(f):
            continue

        # os.mknod(f) # gen.guard will open it
        gen = EnDocGenerator()
        with gen.guard(f):
            gen.module_name = module_name
            gen.api = doc_file
            gen.print_header_reminder()
            gen.print_item()
            api_f.write(doc_file + "\t" + ".. _api_{0}_{1}:\n".format("_".join(
                gen.module_name.split(".")), gen.api))
    api_f.close()


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
            if isinstance(eval(self.module_name + "." + self.api), type):
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

        cls_templates = {
            'default': '''..  autoclass:: {0}.{1}
    :members:
    :inherited-members:
    :noindex:

''',
            'no-inherited': '''..  autoclass:: {0}.{1}
    :members:
    :noindex:

''',
            'fluid.optimizer': '''..  autoclass:: {0}.{1}
    :members:
    :inherited-members:
    :exclude-members: apply_gradients, apply_optimize, backward, load
    :noindex:

'''}
        tmpl = 'default'
        if 'fluid.dygraph' in self.module_name or \
           'paddle.vision' in self.module_name or \
           'paddle.callbacks' in self.module_name or \
           'paddle.hapi.callbacks' in self.module_name or \
           'paddle.io' in self.module_name or \
           'paddle.nn' in self.module_name:
            tmpl = 'no-inherited'
        elif "paddle.optimizer" in self.module_name or \
             "fluid.optimizer" in self.module_name:
            tmpl = 'fluid.optimizer'
        else:
            tmpl = 'default'

        self.stream.write(cls_templates[tmpl].format(self.module_name, self.api))

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
    filter_same_api()

    clean_en_files()
    gen_en_files()
    check_cn_en_match()
