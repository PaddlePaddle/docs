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

# key = id(api), value = list of all apis with the same id.
same_api_map = {}

# read from file './alias_api_mapping'
# key = column 1 (as the real api), value = column 2
alias_api_map = {}

# id
# key = id(api), value = the real api
id_real_api_map = {}

# read from file './not_display_doc_list'
# key = line, value = 1
not_display_doc_map = {}

# read from file './display_doc_list'
# key = line, value = 1
display_doc_map = {}

# the set of all apis
api_set = set()


# walkthrough the paddle package to collect all the apis in api_set
def get_all_api(root_path='paddle'):
    global api_set
    for filefinder, name, ispkg in pkgutil.walk_packages(
            path=paddle.__path__, prefix=paddle.__name__ + '.'):
        # skip the paddle.reader APIs
        if name.startswith("paddle.reader"):
            continue

        try:
            m = eval(name)
        except AttributeError:
            pass
        else:
            if hasattr(m, "__all__"):
                # may have duplication of api
                for api in set(m.__all__):
                    if "," in api:
                        # ?
                        continue

                    # api's fullname
                    api_all = name + "." + api
                    try:
                        fc_id = id(eval(api_all))
                    except AttributeError:
                        pass
                    else:
                        api_set.add(api_all)
    print('collect {} apis.'.format(len(api_set)))

def get_all_same_api():
    global same_api_map, api_set
    for api in api_set:
        fc_id = id(eval(api))
        if fc_id in same_api_map:
            same_api_map[fc_id].append(api)
        else:
            same_api_map[fc_id] = [api]
    print('same_api_map has {} items'.format(len(same_api_map)))

def get_not_display_doc_list(file="./not_display_doc_list"):
    global not_display_doc_map
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            not_display_doc_map[line] = 1
    print('not_display_doc_map has {} items'.format(len(not_display_doc_map)))

def get_display_doc_map(file="./display_doc_list"):
    global display_doc_map
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            display_doc_map[line] = 1
    print('display_doc_map has {} items'.format(len(display_doc_map)))

def get_alias_mapping(file="./alias_api_mapping"):
    global alias_api_map, id_real_api_map
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

    print('id_real_api_map has {} items'.format(len(id_real_api_map)))
    print('alias_api_map has {} items'.format(len(alias_api_map)))

# filter the same_api_map by display list and not display list
def filter_same_api():
    global same_api_map
    for k in list(same_api_map.keys()):
        same_apis = same_api_map[k]
        if is_display_apis(same_apis):
            continue

        if is_not_display_apis(same_apis):
            del same_api_map[k]
    print('filted same_api_map has {} items'.format(len(same_api_map)))

def choose_real_api(api_list):
    global id_real_api_map
    if len(api_list) == 1:
        return api_list[0]

    id_api = id(eval(api_list[0]))
    # the api in id_real_api_map, return real_api
    if id_api in id_real_api_map:
        return id_real_api_map[id_api]
    else:
        # in case of alias_mapping not contain the apis
        # try to find shortest path of api as the real api
        shortest_len = len(api_list[0].split("."))
        shortest_api = api_list[0]
        for x in api_list[1:]:
            len_x = len(x.split("."))
            if len_x < shortest_len:
                shortest_len = len_x
                shortest_api = x

        return shortest_api


# check if any of the same apis in display list
def is_display_apis(api_list):
    global display_doc_map
    for api in api_list:
        if api in display_doc_map:
            return True
    return False


# check api in not_display_list
def is_not_display_apis(api_list):
    global not_display_doc_map
    for api in api_list:
        for key in not_display_doc_map:
            if key == api:
                return True
            # filter all the the module
            if api.startswith(key) and api[len(key)] == '.':
                return True
    return False


def gen_en_files(root_path='paddle', api_label_file="api_label"):
    api_f = open(api_label_file, 'w')

    for api_list in same_api_map.values():
        real_api = choose_real_api(api_list)

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
            'default': '''..  autoclass:: {0}
    :members:
    :inherited-members:
    :noindex:

''',
            'no-inherited': '''..  autoclass:: {0}
    :members:
    :noindex:

''',
            'fluid.optimizer': '''..  autoclass:: {0}
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

        api_full_name = "{}.{}".format(self.module_name, self.api)
        self.stream.write(cls_templates[tmpl].format(api_full_name))

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
