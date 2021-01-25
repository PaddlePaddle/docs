import paddle
import os
import shutil
import time
import pkgutil
import types
import contextlib
import argparse
import json
import sys
import inspect
import ast
"""
generate api_info_dict.json to describe all info about the apis.
"""

en_suffix = "_en.rst"
cn_suffix = "_cn.rst"

# key = id(api), value = dict of api_info{
#   "id":id,
#   "all_names":[],  # all full_names
#   "full_name":"",  # the real name, and the others are the alias name
#   "short_name":"",  # without module name
#   "alias_name":"",  # without module name
#   "module_name":"",  # the module of the real api belongs to
#   "display":True/Flase, # consider the not_display_doc_list and the display_doc_list
#   "has_overwrited_doc":True/False  #
#   "doc_filename"  # document filename without suffix
# }
api_info_dict = {}


# walkthrough the paddle package to collect all the apis in api_set
def get_all_api(root_path='paddle'):
    """
    walk through the paddle package to collect all the apis.
    """
    global api_info_dict
    api_counter = 0
    for filefinder, name, ispkg in pkgutil.walk_packages(
            path=paddle.__path__, prefix=paddle.__name__ + '.'):
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
                    full_name = name + "." + api
                    try:
                        fc_id = id(eval(full_name))
                    except AttributeError:
                        pass
                    else:
                        api_counter += 1
                        if fc_id in api_info_dict:
                            api_info_dict[fc_id]["all_names"].append(full_name)
                        else:
                            api_info_dict[fc_id] = {
                                "all_names": [full_name],
                                "id": fc_id
                            }

    print('collected {} apis, {} distinct apis.'.format(api_counter,
                                                        len(api_info_dict)))


def set_source_code_attrs():
    """
    should has 'full_name' first.
    """
    src_file_start_ind = len(paddle.__path__[0]) - len('paddle/')
    # ast module has end_lineno attr after py 3.8
    has_end_lineno = sys.version_info > (3, 8)
    for id_api in api_info_dict:
        #m = eval(api_info_dict[id_api]['module_name'])
        if api_info_dict[id_api]['module_name'] in sys.modules:
            api_info = api_info_dict[id_api]
            cur_class = sys.modules[api_info['module_name']]
            if hasattr(cur_class, api_info['short_name']):
                # print('processing ', api_info['full_name'])
                api = getattr(cur_class, api_info['short_name'])
                line_no_found = False
                if type(api).__name__ == 'module':
                    module = os.path.splitext(api.__file__)[0] + '.py'
                elif hasattr(api, '__module__') and hasattr(
                        sys.modules[api.__module__], '__file__'):
                    module = os.path.splitext(sys.modules[api.__module__]
                                              .__file__)[0] + '.py'
                    if os.path.isfile(module):
                        with open(module) as module_file:
                            module_ast = ast.parse(module_file.read())

                            node_definition = ast.ClassDef if inspect.isclass(
                                api) else ast.FunctionDef
                            for node in module_ast.body:
                                if isinstance(
                                        node, node_definition
                                ) and node.name == api_info['short_name']:
                                    line_no = node.lineno
                                    line_no_found = True
                                    if has_end_lineno:
                                        end_line_no = node.end_lineno
                                    break

                            # If we could not find it, we look at assigned objects.
                            if not line_no_found:
                                for node in module_ast.body:
                                    if isinstance(
                                            node, ast.Assign) and api_info[
                                                'short_name'] in [
                                                    target.id
                                                    for target in node.targets
                                                ]:
                                        line_no = node.lineno
                                        line_no_found = True
                                        if has_end_lineno:
                                            end_line_no = node.end_lineno
                                        break
                if line_no_found:
                    api_info_dict[id_api]["lineno"] = line_no
                    if has_end_lineno:
                        api_info_dict[id_api]["end_lineno"] = end_line_no
                if len(module) > src_file_start_ind:
                    api_info_dict[id_api]["src_file"] = module[
                        src_file_start_ind:]
                else:
                    api_info_dict[id_api]["src_file"] = module


def set_display_attr_of_apis():
    """
    set the display attr
    """
    display_none_apis = set(
        [line.strip() for line in open("./not_display_doc_list", "r")])
    display_yes_apis = set(
        [line.strip() for line in open("./display_doc_list", "r")])
    print('display_none_apis has {} items, display_yes_apis has {} items'.
          format(len(display_none_apis), len(display_yes_apis)))
    # file the same apis
    for id_api in api_info_dict:
        all_names = api_info_dict[id_api]["all_names"]
        display_yes = False
        for n in all_names:
            if n in display_yes_apis:
                display_yes = True
                break
        if display_yes:
            api_info_dict[id_api]["display"] = True
        else:
            display_yes = True
            for n in all_names:
                for dn in display_none_apis:
                    if n.startswith(dn):
                        display_yes = False
                        break
                if not display_yes:
                    break
            if not display_yes:
                api_info_dict[id_api]["display"] = False


def set_real_api_alias_attr():
    """
    set the full_name,alias attr and so on.
    """
    for line in open("./alias_api_mapping", "r"):
        linecont = line.strip()
        lineparts = linecont.split()
        if len(lineparts) < 2:
            print('line "', line, '" splited to ', lineparts)
            continue
        try:
            real_api = lineparts[0]
            m = eval(real_api)
        except AttributeError:
            print("AttributeError:", real_api)
        else:
            api_id = id(m)
            if api_id in api_info_dict:
                api_info_dict[api_id]["alias_name"] = lineparts[1]
                api_info_dict[api_id]["full_name"] = lineparts[0]
            pass

    for api_id in api_info_dict:
        if "full_name" not in api_info_dict[api_id]:
            api_info_dict[api_id]["full_name"] = get_shortest_api(
                api_info_dict[api_id]["all_names"])

    for api_id in api_info_dict:
        real_api = api_info_dict[api_id]["full_name"]
        real_api_parts = real_api.split(".")
        api_info_dict[api_id]["module_name"] = ".".join(real_api_parts[0:-1])
        api_info_dict[api_id]["doc_filename"] = real_api.replace(".", "/")
        api_info_dict[api_id]["short_name"] = real_api_parts[-1]


def get_shortest_api(api_list):
    """
    find the shortest api in list.
    """
    if len(api_list) == 1:
        return api_list[0]
    # try to find shortest path of api as the real api
    shortest_len = len(api_list[0].split("."))
    shortest_api = api_list[0]
    for x in api_list[1:]:
        len_x = len(x.split("."))
        if len_x < shortest_len:
            shortest_len = len_x
            shortest_api = x

    return shortest_api


def remove_all_en_files(path="./paddle"):
    """
    remove all the existed en doc files
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(en_suffix):
                os.remove(os.path.join(root, file))


def gen_en_files(api_label_file="api_label"):
    """
    generate all the en doc files.
    """
    with open(api_label_file, 'w') as api_label:
        for id_api, api_info in api_info_dict.items():
            # api_info = api_info_dict[id_api]
            if "display" in api_info and not api_info["display"]:
                continue
            path = os.path.dirname(api_info["doc_filename"])
            if not os.path.exists(path):
                os.makedirs(path)
            f = api_info["doc_filename"] + en_suffix
            if os.path.exists(f):
                continue
            gen = EnDocGenerator()
            with gen.guard(f):
                gen.module_name = api_info["module_name"]
                gen.api = api_info["short_name"]
                gen.print_header_reminder()
                gen.print_item()
                api_label.write("{1}\t.. _api_{0}_{1}:\n".format("_".join(
                    gen.module_name.split(".")), gen.api))


def check_cn_en_match(path="./paddle", diff_file="en_cn_files_diff"):
    """
    skip
    """
    osp_join = os.path.join
    osp_exists = os.path.exists
    with open(diff_file, 'w') as fo:
        tmpl = "{}\t{}\n"
        fo.write(tmpl.format("exist", "not_exits"))
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(en_suffix):
                    cf = file.replace(en_suffix, cn_suffix)
                    if not osp_exists(osp_join(root, cf)):
                        fo.write(
                            tmpl.format(
                                osp_join(root, file), osp_join(root, cf)))
                elif file.endswith(cn_suffix):
                    ef = file.replace(cn_suffix, en_suffix)
                    if not osp_exists(osp_join(root, ef)):
                        fo.write(
                            tmpl.format(
                                osp_join(root, file), osp_join(root, ef)))


class EnDocGenerator(object):
    """
    skip
    """

    def __init__(self, name=None, api=None):
        """
        init
        """
        self.module_name = name
        self.api = api
        self.stream = None

    @contextlib.contextmanager
    def guard(self, filename):
        """
        open the file
        """
        assert self.stream is None, "stream must be None"
        self.stream = open(filename, 'w')
        yield
        self.stream.close()
        self.stream = None

    def print_item(self):
        """
        as name
        """
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
        """
        as name
        """
        self.stream.write('''..  THIS FILE IS GENERATED BY `gen_doc.{py|sh}`
    !DO NOT EDIT THIS FILE MANUALLY!

''')

    def _print_ref_(self):
        """
        as name
        """
        self.stream.write(".. _api_{0}_{1}:\n\n".format("_".join(
            self.module_name.split(".")), self.api))

    def _print_header_(self, name, dot, is_title):
        """
        as name
        """
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
        """
        as name
        """
        self._print_ref_()
        self._print_header_(self.api, dot='-', is_title=False)

        cls_templates = {
            'default':
            '''..  autoclass:: {0}
    :members:
    :inherited-members:
    :noindex:

''',
            'no-inherited':
            '''..  autoclass:: {0}
    :members:
    :noindex:

''',
            'fluid.optimizer':
            '''..  autoclass:: {0}
    :members:
    :inherited-members:
    :exclude-members: apply_gradients, apply_optimize, backward, load
    :noindex:

'''
        }
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
        """
        as name
        """
        self._print_ref_()
        self._print_header_(self.api, dot='-', is_title=False)
        self.stream.write('''..  autofunction:: {0}.{1}
    :noindex:

'''.format(self.module_name, self.api))


if __name__ == "__main__":
    get_all_api()
    set_display_attr_of_apis()
    set_real_api_alias_attr()
    set_source_code_attrs()
    json.dump(api_info_dict, open("api_info_dict.json", "w"), indent=4)
    gen_en_files()
    check_cn_en_match()
