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
import logging
import importlib
import re
import subprocess
import multiprocessing
import platform
import extract_api_from_docs
from queue import Queue

"""
generate api_info_dict.json to describe all info about the apis.
"""

en_suffix = "_en.rst"
cn_suffix = "_cn.rst"
NOT_DISPLAY_DOC_LIST_FILENAME = "./not_display_doc_list"
DISPLAY_DOC_LIST_FILENAME = "./display_doc_list"
CALLED_APIS_IN_THE_DOCS = (
    './called_apis_from_docs.json'  # in the guides and tutorials documents
)
SAMPLECODE_TEMPDIR = './sample-codes'
RUN_ON_DEVICE = "cpu"
EQUIPPED_DEVICES = set(['cpu'])
GPU_ID = 0

# key = id(api), value = dict of api_info{
#   "id":id,
#   "all_names":[],  # all full_names
#   "full_name":"",  # the real name, and the others are the alias name
#   "short_name":"",  # without module name
#   "module_name":"",  # the module of the real api belongs to
#   "display":True/Flase, # consider the not_display_doc_list and the display_doc_list
#   "has_overwrited_doc":True/False  #
#   "doc_filename"  # document filename without suffix
#   "suggested_name":"",  # the shortest name in all_names
# }
api_info_dict = {}
parsed_mods = {}
referenced_from_apis_dict = {}
referenced_from_file_titles = {}

logger = logging.getLogger()
if logger.handlers:
    # we assume the first handler is the one we want to configure
    console = logger.handlers[0]
else:
    console = logging.StreamHandler()
    logger.addHandler(console)
console.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
    )
)


# step 1: walkthrough the paddle package to collect all the apis in api_set
def get_all_api(root_path='paddle', attr="__all__"):
    """
    walk through the paddle package to collect all the apis.
    """
    global api_info_dict
    api_counter = 0
    for filefinder, name, ispkg in pkgutil.walk_packages(
        path=paddle.__path__, prefix=paddle.__name__ + '.'
    ):
        try:
            if name in sys.modules:
                m = sys.modules[name]
            else:
                # importlib.import_module(name)
                m = eval(name)
                continue
        except AttributeError:
            logger.warning("AttributeError occurred when `eval(%s)`", name)
            pass
        else:
            api_counter += process_module(m, attr)

    api_counter += process_module(paddle, attr)

    logger.info(
        '%s: collected %d apis, %d distinct apis.',
        attr,
        api_counter,
        len(api_info_dict),
    )


def insert_api_into_dict(full_name, gen_doc_anno=None):
    """
    insert add api into the api_info_dict

    Return:
        api_info object or None
    """
    try:
        obj = eval(full_name)
        fc_id = id(obj)
    except AttributeError:
        logger.warning("AttributeError occurred when `id(eval(%s))`", full_name)
        return None
    except:
        logger.warning("Exception occurred when `id(eval(%s))`", full_name)
        return None
    else:
        logger.debug("adding %s to api_info_dict.", full_name)
        if fc_id in api_info_dict:
            api_info_dict[fc_id]["all_names"].add(full_name)
        else:
            api_info_dict[fc_id] = {
                "all_names": set([full_name]),
                "id": fc_id,
                "object": obj,
                "type": type(obj).__name__,
            }
            docstr = inspect.getdoc(obj)
            if docstr:
                api_info_dict[fc_id]["docstring"] = inspect.cleandoc(docstr)
            if gen_doc_anno:
                api_info_dict[fc_id]["gen_doc_anno"] = gen_doc_anno
        return api_info_dict[fc_id]


# step 1 fill field : `id` & `all_names`, type, docstring
def process_module(m, attr="__all__"):
    api_counter = 0
    if hasattr(m, attr):
        # may have duplication of api
        for api in set(getattr(m, attr)):
            if api[0] == '_':
                continue
            # Exception occurred when `id(eval(paddle.dataset.conll05.test, get_dict))`
            if ',' in api:
                continue

            # api's fullname
            full_name = m.__name__ + "." + api
            api_info = insert_api_into_dict(full_name)
            if api_info is not None:
                api_counter += 1
                if inspect.isclass(api_info['object']):
                    for name, value in inspect.getmembers(api_info['object']):
                        if not name.startswith("_"):
                            try:
                                method_full_name = (
                                    full_name + '.' + name
                                )  # value.__name__
                                if (
                                    name
                                    and value
                                    and isinstance(value, property)
                                ):
                                    method_api_info = insert_api_into_dict(
                                        method_full_name, 'class_property'
                                    )
                                    if method_api_info is not None:
                                        api_counter += 1
                                elif hasattr(value, '__name__'):
                                    method_api_info = insert_api_into_dict(
                                        method_full_name, 'class_method'
                                    )
                                    if method_api_info is not None:
                                        api_counter += 1
                            except ValueError as e:
                                logger.error(
                                    'ValueError when processing %s: %s',
                                    method_full_name,
                                    str(e),
                                )
    return api_counter


# step 3 fill field : args, src_file, lineno, end_lineno, short_name, full_name, module_name, doc_filename
def set_source_code_attrs():
    """
    should has 'full_name' first.
    """
    src_file_start_ind = len(paddle.__path__[0]) - len('paddle/')
    # ast module has end_lineno attr after py 3.8

    for id_api in api_info_dict:
        item = api_info_dict[id_api]
        obj = item["object"]
        obj_type_name = item["type"]
        logger.debug("processing %s:%s:%s", obj_type_name, item["id"], str(obj))
        if obj_type_name == "module":
            if (
                hasattr(obj, '__file__')
                and obj.__file__ is not None
                and len(obj.__file__) > src_file_start_ind
            ):
                api_info_dict[id_api]["src_file"] = obj.__file__[
                    src_file_start_ind:
                ]
            parse_module_file(obj)
            api_info_dict[id_api]["full_name"] = obj.__name__
            api_info_dict[id_api]["package"] = obj.__package__
            api_info_dict[id_api]["short_name"] = split_name(obj.__name__)[1]
        elif hasattr(obj, '__module__') and obj.__module__ in sys.modules:
            mod_name = obj.__module__
            mod = sys.modules[mod_name]
            parse_module_file(mod)
        else:
            if hasattr(obj, '__name__'):
                mod_name, short_name = split_name(obj.__name__)
                if mod_name in sys.modules:
                    mod = sys.modules[mod_name]
                    parse_module_file(mod)
                else:
                    logger.debug(
                        "{}, {}, {}".format(
                            item["id"], item["type"], item["all_names"]
                        )
                    )
            else:
                found = False
                for name in item["all_names"]:
                    mod_name, short_name = split_name(name)
                    if mod_name in sys.modules:
                        mod = sys.modules[mod_name]
                        parse_module_file(mod)
                        found = True
                if not found:
                    logger.debug(
                        "{}, {}, {}".format(
                            item["id"], item["type"], item["all_names"]
                        )
                    )


def split_name(name):
    try:
        r = name.rindex('.')
        return [name[:r], name[r + 1 :]]
    except:
        return ['', name]


def parse_module_file(mod):
    skip_this_mod = False
    if mod in parsed_mods:
        skip_this_mod = True
    if skip_this_mod:
        return
    else:
        parsed_mods[mod] = True

    src_file_start_ind = len(paddle.__path__[0]) - len('paddle/')
    has_end_lineno = sys.version_info > (3, 8)
    if hasattr(mod, '__name__') and hasattr(mod, '__file__'):
        src_file = mod.__file__
        mod_name = mod.__name__
        if not (isinstance(src_file, str) and isinstance(src_file, str)):
            logger.error(
                '%s: mod_name=%s, src_file=%s', str(mod), mod_name, src_file
            )
            return
        logger.debug("parsing %s:%s", mod_name, src_file)
        if len(mod_name) >= 6 and mod_name[:6] == 'paddle':
            fn_splited = os.path.splitext(src_file)
            if len(fn_splited) > 1 and fn_splited[1].lower() == '.py':
                mod_ast = ast.parse(open(src_file, "r").read())
                for node in mod_ast.body:
                    short_names = []
                    if (
                        (
                            isinstance(node, ast.ClassDef)
                            or isinstance(node, ast.FunctionDef)
                        )
                        and hasattr(node, 'name')
                        and hasattr(sys.modules[mod_name], node.name)
                        and node.name[0] != '_'
                    ):
                        short_names.append(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if hasattr(target, 'id') and target.id[0] != '_':
                                short_names.append(target.id)
                    else:
                        pass
                    for short_name in short_names:
                        obj_full_name = mod_name + '.' + short_name
                        logger.debug("processing %s", obj_full_name)
                        try:
                            obj_this = eval(obj_full_name)
                            obj_id = id(obj_this)
                        except:
                            logger.warning(
                                "%s maybe %s.%s",
                                obj_full_name,
                                mod.__package__,
                                short_name,
                            )
                            obj_full_name = mod.__package__ + '.' + short_name
                            try:
                                obj_this = eval(obj_full_name)
                                obj_id = id(obj_this)
                            except:
                                continue
                        if (
                            obj_id in api_info_dict
                            and "lineno" not in api_info_dict[obj_id]
                        ):
                            api_info_dict[obj_id]["src_file"] = src_file[
                                src_file_start_ind:
                            ]
                            api_info_dict[obj_id][
                                "doc_filename"
                            ] = obj_full_name.replace('.', '/')
                            api_info_dict[obj_id]["full_name"] = obj_full_name
                            api_info_dict[obj_id]["short_name"] = short_name
                            api_info_dict[obj_id]["module_name"] = mod_name
                            api_info_dict[obj_id]["lineno"] = node.lineno
                            if has_end_lineno:
                                api_info_dict[obj_id][
                                    "end_lineno"
                                ] = node.end_lineno
                            if isinstance(node, ast.FunctionDef):
                                api_info_dict[obj_id][
                                    "args"
                                ] = gen_functions_args_str(node)
                            elif isinstance(node, ast.ClassDef):
                                for n in node.body:
                                    if (
                                        hasattr(n, 'name')
                                        and n.name == '__init__'
                                    ):
                                        api_info_dict[obj_id][
                                            "args"
                                        ] = gen_functions_args_str(n)
                                        break
                        else:
                            logger.debug("%s omitted", obj_full_name)
            else:  # pybind11 ...
                for short_name in mod.__dict__:
                    if short_name[0] != '_':
                        obj_full_name = mod_name + '.' + short_name
                        logger.debug("processing %s", obj_full_name)
                        try:
                            obj_this = eval(obj_full_name)
                            obj_id = id(obj_this)
                        except:
                            logger.warning("%s eval error", obj_full_name)
                            continue
                        if (
                            obj_id in api_info_dict
                            and "lineno" not in api_info_dict[obj_id]
                        ):
                            api_info_dict[obj_id]["src_file"] = src_file[
                                src_file_start_ind:
                            ]
                            api_info_dict[obj_id]["full_name"] = obj_full_name
                            api_info_dict[obj_id]["short_name"] = short_name
                            api_info_dict[obj_id]["module_name"] = mod_name
                            api_info_dict[obj_id][
                                "doc_filename"
                            ] = obj_full_name.replace('.', '/')
                        else:
                            logger.debug("%s omitted", obj_full_name)


def gen_functions_args_str(node):
    str_args_list = []
    if isinstance(node, ast.FunctionDef):
        # 'args', 'defaults', 'kw_defaults', 'kwarg', 'kwonlyargs', 'posonlyargs', 'vararg'
        for arg in node.args.args:
            if not arg.arg == 'self':
                str_args_list.append(arg.arg)

        defarg_ind_start = len(str_args_list) - len(node.args.defaults)
        for defarg_ind in range(len(node.args.defaults)):
            if isinstance(node.args.defaults[defarg_ind], ast.Name):
                str_args_list[defarg_ind_start + defarg_ind] += '=' + str(
                    node.args.defaults[defarg_ind].id
                )
            elif isinstance(node.args.defaults[defarg_ind], ast.Constant):
                defarg_val = str(node.args.defaults[defarg_ind].value)
                if isinstance(node.args.defaults[defarg_ind].value, str):
                    defarg_val = f"'{defarg_val}'"
                str_args_list[defarg_ind_start + defarg_ind] += '=' + defarg_val
        if node.args.vararg is not None:
            str_args_list.append('*' + node.args.vararg.arg)
        if len(node.args.kwonlyargs) > 0:
            if node.args.vararg is None:
                str_args_list.append('*')
            for kwoarg, d in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if isinstance(d, ast.Constant):
                    str_args_list.append("{}={}".format(kwoarg.arg, d.value))
                elif isinstance(d, ast.Name):
                    str_args_list.append("{}={}".format(kwoarg.arg, d.id))
        if node.args.kwarg is not None:
            str_args_list.append('**' + node.args.kwarg.arg)

    return ', '.join(str_args_list)


# step 2 fill field : `display`
def set_display_attr_of_apis():
    """
    set the display attr
    """
    if os.path.exists(NOT_DISPLAY_DOC_LIST_FILENAME):
        display_none_apis = set(
            [line.strip() for line in open(NOT_DISPLAY_DOC_LIST_FILENAME, "r")]
        )
    else:
        logger.warning("file not exists: %s", NOT_DISPLAY_DOC_LIST_FILENAME)
        display_none_apis = set()
    if os.path.exists(DISPLAY_DOC_LIST_FILENAME):
        display_yes_apis = set(
            [line.strip() for line in open(DISPLAY_DOC_LIST_FILENAME, "r")]
        )
    else:
        logger.warning("file not exists: %s", DISPLAY_DOC_LIST_FILENAME)
        display_yes_apis = set()
    logger.info(
        'display_none_apis has %d items, display_yes_apis has %d items',
        len(display_none_apis),
        len(display_yes_apis),
    )

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
                logger.info("set {} display to False".format(id_api))


def check_module_in_black_list(module_name):
    black_module_list = [
        'paddle.fluid',
    ]
    for i in black_module_list:
        if i in module_name:
            return True
    return False


def get_all_modules():
    """
    get all modules from paddle
    :return: module list
    """
    module_str_queue = Queue()
    module_str_queue.put('paddle')

    MODULE_CLS = type(paddle)
    module_list = []
    while not module_str_queue.empty():
        module_name = module_str_queue.get()
        try:
            module = importlib.import_module(module_name)
            module_list.append(module)
            for sub_module_str in dir(module):
                if sub_module_str.startswith('_'):
                    continue
                full_sub_module_path = '.'.join([module_name, sub_module_str])
                sub_module = eval(full_sub_module_path)
                if isinstance(sub_module, MODULE_CLS):
                    module_str_queue.put(full_sub_module_path)
        except Exception as e:
            continue

    return module_list


def get_public_modules():
    """
    get public modules from paddle
    :return: module list
    """
    public_module_list = []
    all_modules = get_all_modules()
    for module in all_modules:
        if check_module_in_black_list(module.__name__):
            logger.info('module %s in black module list', module.__name__)
            continue
        if hasattr(module, '__all__'):
            api_in_module = module.__all__
            if len(api_in_module) == 0:
                logger.info('API in module %s is empty', module.__name__)
                continue
            public_module_list.append(module)
    return public_module_list


def get_api_from_module(module):
    """
    get api list from module
    :param module: module object
    :return: api list
    """
    if not hasattr(module, '__all__'):
        return []
    return module.__all__


def set_api_sketch():
    """
    set the in_api_sktech attr. may replace the set_display_attr_of_apis.
    """
    global api_info_dict
    modulelist = get_public_modules()

    alldict = {}
    for module in modulelist:
        if hasattr(module, '__all__'):
            old_all = module.__all__
        else:
            old_all = []
            dirall = dir(module)
            for item in dirall:
                if item.startswith('__'):
                    continue
                old_all.append(item)
        alldict.update({module.__name__: old_all})

    old_all = []
    dirall = dir(paddle.Tensor)
    for item in dirall:
        if item.startswith('_'):
            continue
        old_all.append(item)
    alldict.update({'paddle.Tensor': old_all})

    all_api_found = {}
    for m, apis in alldict.items():
        for api in apis:
            all_api_found['{}.{}'.format(m, api)] = False

    for api in all_api_found.keys():
        for id_api in api_info_dict.keys():
            if ('all_names' in api_info_dict[id_api]) and (
                api in api_info_dict[id_api]['all_names']
            ):
                all_api_found[api] = True
                api_info_dict[id_api]['in_api_sketch'] = True
                if 'api_sketch_names' not in api_info_dict[id_api]:
                    api_info_dict[id_api]['api_sketch_names'] = []
                api_info_dict[id_api]['api_sketch_names'].append(api)
                break

    api_not_in_dict = [api for api in all_api_found if not all_api_found[api]]
    if api_not_in_dict:
        logger.warning(
            "some apis are not in api_info_dict: %s", str(api_not_in_dict)
        )


# step fill field: referenced_from
def set_referenced_from_attr():
    """
    set the referenced_from field.

    values are the guides and tutorial documents.
    """
    global api_info_dict
    global referenced_from_apis_dict, referenced_from_file_titles
    if (
        len(referenced_from_apis_dict) > 0
        and len(referenced_from_file_titles) > 0
    ):
        apis_refers = referenced_from_apis_dict
        rev_apis_refers = {}
        for docfn in apis_refers:
            for api in apis_refers[docfn]:
                if api in rev_apis_refers:
                    rev_apis_refers[api].append(docfn)
                else:
                    rev_apis_refers[api] = [docfn]
        for api in rev_apis_refers:
            try:
                m = eval(api)
            except AttributeError:
                logger.warning("AttributeError: %s", api)
            except NameError:
                logger.warning("NameError: %s", api)
            else:
                api_id = id(m)
                if api_id in api_info_dict:
                    ref_from = []
                    for a in rev_apis_refers[api]:
                        ref_from.append(
                            {
                                'file': a,
                                'title': referenced_from_file_titles[a]
                                if a in referenced_from_file_titles
                                else '',
                            }
                        )
                    api_info_dict[api_id]["referenced_from"] = ref_from
                else:
                    logger.warning(
                        "%s (id:%d) not in the api_info_dict.", api, api_id
                    )


def collect_referenced_from_infos(docdirs):
    """
    collect all the referenced_from infos from ../guides and ../tutorial
    """
    global referenced_from_apis_dict, referenced_from_file_titles
    (
        referenced_from_apis_dict,
        referenced_from_file_titles,
    ) = extract_api_from_docs.extract_all_infos(docdirs)


def remove_all_en_files(path="./paddle"):
    """
    remove all the existed en doc files
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(en_suffix):
                os.remove(os.path.join(root, file))


def is_Tensor_method(api_info):
    for n in api_info['all_names']:
        if n.startswith('paddle.Tensor'):
            return True
    return False


# using `doc_filename`
def gen_en_files(api_label_file="api_label"):
    """
    generate all the en doc files.
    """
    with open(api_label_file, 'w') as api_label:
        for id_api, api_info in api_info_dict.items():
            # api_info = api_info_dict[id_api]
            if 'full_name' in api_info and api_info['full_name'].endswith(
                'Overview'
            ):
                continue
            elif "display" in api_info and not api_info["display"]:
                logger.debug("{} display False".format(id_api))
                continue
            elif 'type' in api_info and api_info['type'] in [
                'module',
                'method',
                'VarType',
                'builtin_function_or_method',
                'dict',
                'float',
                'str',
            ]:
                continue
            elif (
                'gen_doc_anno' in api_info
                and api_info['gen_doc_anno'] == 'class_method'
                and (not is_Tensor_method(api_info))
            ):
                continue
            elif "doc_filename" not in api_info:
                logger.debug(
                    "{} does not have doc_filename field.".format(id_api)
                )
                continue
            else:
                logger.debug(api_info["doc_filename"])
            path = os.path.dirname(api_info["doc_filename"])
            if not os.path.exists(path):
                os.makedirs(path)
            f = api_info["doc_filename"] + en_suffix
            if os.path.exists(f):
                continue
            gen = EnDocGenerator(api_info)
            api_name, api_ref_name = gen()
            if api_name and api_ref_name:
                api_label.write("{}\t.. {}:\n".format(api_name, api_ref_name))


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
                                osp_join(root, file), osp_join(root, cf)
                            )
                        )
                elif file.endswith(cn_suffix):
                    ef = file.replace(cn_suffix, en_suffix)
                    if not osp_exists(osp_join(root, ef)):
                        fo.write(
                            tmpl.format(
                                osp_join(root, file), osp_join(root, ef)
                            )
                        )


class EnDocGenerator:
    """
    skip
    """

    def __init__(self, api_info):
        """
        init
        """
        self.api_info = api_info
        if 'suggested_name' in self.api_info:
            self.api_name = self.api_info['suggested_name']
        elif 'full_name' in self.api_info:
            self.api_name = self.api_info['full_name']
        else:
            logger.warning(
                "%s has no attr called full_name/suggested_name",
                str(self.api_info),
            )
            self.api_name = None
        self.api_ref_name = (
            '_api_' + self.api_name.replace('.', '_') if self.api_name else None
        )
        # disarding the api_info['short_name'], cause it may be different.
        _, self.short_name = split_name(self.api_name)
        self.stream = None
        self.object = None

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
        if isinstance(self.object, type):
            self.print_class()
        elif isinstance(self.object, types.FunctionType):
            self.print_function()
        else:
            logger.warning(
                "%s: not supported type %s",
                str(self.api_name),
                type(self.object),
            )

    def print_header_reminder(self):
        """
        as name
        """
        self.stream.write(
            '''..  THIS FILE IS GENERATED BY `gen_doc.{py|sh}`
    !DO NOT EDIT THIS FILE MANUALLY!

'''
        )

    def _print_ref_(self):
        """
        as name
        """
        if self.api_name is None:
            return
        self.stream.write(".. {}:\n\n".format(self.api_ref_name))

    def _print_header_(self, name, dot, is_title):
        """
        as name
        """
        mo = re.match(r'^(.*?)(_+)$', name)
        if mo:
            name = mo.group(1) + r'\_' * len(mo.group(2))
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
        self._print_header_(self.short_name, dot='-', is_title=False)

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

''',
        }
        tmpl = 'default'
        for m in [
            'fluid.dygraph',
            'paddle.vision',
            'paddle.callbacks',
            'paddle.hapi.callbacks',
            'paddle.io',
            'paddle.nn',
            'paddle.incubate.nn',
            'paddle.audio',
        ]:
            if self.api_name.startswith(m):
                tmpl = 'no-inherited'
        if tmpl == 'default':
            for m in ["paddle.optimizer", "fluid.optimizer"]:
                if self.api_name.startswith(m):
                    tmpl = 'fluid.optimizer'

        self.stream.write(cls_templates[tmpl].format(self.api_name))

    def print_function(self):
        """
        as name
        """
        self._print_ref_()
        self._print_header_(self.short_name, dot='-', is_title=False)
        self.stream.write(
            '''..  autofunction:: {}
    :noindex:

'''.format(
                self.api_name
            )
        )

    def __call__(self):
        """
        generate the rst file.
        """
        try:
            if 'object' in self.api_info:
                self.object = self.api_info['object']
            elif self.api_name is not None:
                self.object = eval(self.api_name)
            else:
                logger.warning(
                    "%s has no attr called object/full_name/suggested_name",
                    str(self.api_info),
                )
                return None, None
        except AttributeError:
            logger.warning("attribute error for %s ", str(self.api_info))
            return None, None
        else:
            if (not isinstance(self.object, type)) and (
                not isinstance(self.object, types.FunctionType)
            ):
                logger.warning(
                    "%s: not supported type %s",
                    str(self.api_name),
                    type(self.object),
                )
                return None, None
        if self.api_name:
            filename = self.api_info['doc_filename'] + en_suffix
            with self.guard(filename):
                self.print_header_reminder()
                self.print_item()
        return self.api_name, self.api_ref_name


def get_shortest_api(api_list):
    """
    find the shortest api name (suggested name) in list.

    Problems:
    1. fuild - if there is any apis don't contain 'fluid' in name, use them.
    2. core vs core_avx - using the 'core'.
    """
    if len(api_list) == 1:
        return api_list[0]
    # try to find shortest path of api as the real api
    api_info = (
        []
    )  # {'name': name, 'fluid_in_name': True/False, 'core_avx_in_name': True/Flase', 'len': len}
    for api in api_list:
        fields = api.split('.')
        api_info.append(
            {
                'name': api,
                'fluid_in_name': 'fluid' in fields,
                'core_avx_in_name': 'core_avx' in fields,
                'len': len(fields),
            }
        )

    def shortest(api_info):
        if not api_info:
            return None
        elif len(api_info) == 1:
            return api_info[0].get('name')
        api_info.sort(key=lambda ele: ele.get('len'))
        return api_info[0].get('name')

    if not all([api.get('fuild_in_name') for api in api_info]):
        api_info = [api for api in api_info if not api.get('fluid_in_name')]
    sn = shortest([api for api in api_info if not api.get('core_avx_in_name')])
    if sn is None:
        sn = shortest(api_info)
    return sn


def insert_suggested_names():
    """
    add suggested_name field, updte the doc_filename, and sort the all_names and api_sketch_names.
    """
    pat = re.compile(r'paddle\.fluid\.core_[\w\d]+\.(.*)$')

    def sort_name_list(api_names):
        """
        sort and move paddle.Tensor.* to the end
        """
        names_sorted = sorted(list(api_names))
        cnt = 0  # count of paddle.Tensor.*
        for n in names_sorted:
            if n.startswith('paddle.Tensor.'):
                cnt += 1
            else:
                break
        if cnt:
            names_sorted = names_sorted[cnt:] + names_sorted[:cnt]
        return names_sorted

    for id_api in api_info_dict:
        if "all_names" not in api_info_dict[id_api]:
            api_info_dict[id_api]["all_names"] = set()
        if (
            "full_name" in api_info_dict[id_api]
            and api_info_dict[id_api]["full_name"]
            not in api_info_dict[id_api]["all_names"]
        ):
            api_info_dict[id_api]["all_names"].add(
                api_info_dict[id_api]["full_name"]
            )
        for n in list(api_info_dict[id_api]["all_names"]):
            # paddle.fluid.core_avx.* -> paddle.fluid.core.*
            mo = pat.match(n)
            if mo:
                api_info_dict[id_api]["all_names"].add(
                    'paddle.fluid.core.' + mo.group(1)
                )
        api_info_dict[id_api]["all_names"] = sort_name_list(
            api_info_dict[id_api]["all_names"]
        )
        sn = None
        if 'api_sketch_names' in api_info_dict[id_api]:
            api_info_dict[id_api]['api_sketch_names'] = sort_name_list(
                api_info_dict[id_api]['api_sketch_names']
            )
            sn = get_shortest_api(api_info_dict[id_api]['api_sketch_names'])
        if not sn:
            sn = get_shortest_api(api_info_dict[id_api]["all_names"])
        if sn:
            # Delete alias_name, api_info_dict[id_api]["alias_name"] = sn
            api_info_dict[id_api]["suggested_name"] = sn
            api_info_dict[id_api]["doc_filename"] = sn.replace('.', '/')


def filter_out_object_of_api_info_dict():
    """
    filter out the object before dump json string.
    """
    for id_api in api_info_dict:
        if 'object' in api_info_dict[id_api]:
            del api_info_dict[id_api]['object']


def strip_ps1_from_codeblock(codeblock):
    """strip PS1(>>> ) from codeblock"""
    match_obj = re.search(r"\n>>>\s?", "\n" + codeblock)
    if match_obj is None:
        return codeblock

    codeblock_clean = []
    for line in codeblock.splitlines():
        match_obj = re.match(r"^>>>\s?", line.lstrip())
        if match_obj is None:
            codeblock_clean.append("# {}".format(line))
        else:
            codeblock_clean.append(line[match_obj.end() :])

    return "\n".join(codeblock_clean)


def extract_code_blocks_from_docstr(docstr, google_style=True):
    """
    extract code-blocks from the given docstring.
    DON'T include the multiline-string definition in code-blocks.
    The *Examples* section must be the last.
    Args:
        docstr(str): docstring
        google_style(bool): if not use google_style, the code blocks will be extracted from all the parts of docstring.
    Return:
        code_blocks: A list of code-blocks, indent removed.
                     element {'name': the code-block's name, 'id': sequence id.
                              'codes': codes, 'required': 'gpu', 'in_examples': bool, code block in `Examples` or not,}
    """
    code_blocks = []

    mo = re.search(r"Examples?:", docstr)

    if google_style and mo is None:
        return code_blocks

    example_start = len(docstr) if mo is None else mo.start()
    docstr_describe = docstr[:example_start].splitlines()
    docstr_examples = docstr[example_start:].splitlines()

    docstr_list = []
    if google_style:
        example_lineno = 0
        docstr_list = docstr_examples
    else:
        example_lineno = len(docstr_describe)
        docstr_list = docstr_describe + docstr_examples

    lastlineindex = len(docstr_list) - 1

    cb_start_pat = re.compile(r"code-block::\s*python")
    cb_param_pat = re.compile(r"^\s*:(\w+):\s*(\S*)\s*$")
    cb_required_pat = re.compile(r"^\s*#\s*require[s|d]\s*:\s*(\S+)\s*$")

    cb_info = {}
    cb_info['cb_started'] = False
    cb_info['cb_cur'] = []
    cb_info['cb_cur_indent'] = -1
    cb_info['cb_cur_name'] = None
    cb_info['cb_cur_seq_id'] = 0
    cb_info['cb_required'] = None

    def _cb_started():
        # nonlocal cb_started, cb_cur_name, cb_required, cb_cur_seq_id
        cb_info['cb_started'] = True
        cb_info['cb_cur_seq_id'] += 1
        cb_info['cb_cur_name'] = None
        cb_info['cb_required'] = None

    def _append_code_block(in_examples):
        # nonlocal code_blocks, cb_cur, cb_cur_name, cb_cur_seq_id, cb_required
        code_blocks.append(
            {
                'codes': strip_ps1_from_codeblock(
                    inspect.cleandoc("\n".join(cb_info['cb_cur']))
                ),
                'name': cb_info['cb_cur_name'],
                'id': cb_info['cb_cur_seq_id'],
                'required': cb_info['cb_required'],
                'in_examples': in_examples,
            }
        )

    for lineno, linecont in enumerate(docstr_list):
        if re.search(cb_start_pat, linecont):
            if not cb_info['cb_started']:
                _cb_started()
                continue
            else:
                # cur block end
                if len(cb_info['cb_cur']):
                    _append_code_block(lineno > example_lineno)
                _cb_started()  # another block started
                cb_info['cb_cur_indent'] = -1
                cb_info['cb_cur'] = []
        else:
            if cb_info['cb_started']:
                # handle the code-block directive's options
                mo_p = cb_param_pat.match(linecont)
                if mo_p:
                    if mo_p.group(1) == 'name':
                        cb_info['cb_cur_name'] = mo_p.group(2)
                    continue
                # read the required directive
                mo_r = cb_required_pat.match(linecont)
                if mo_r:
                    cb_info['cb_required'] = mo_r.group(1)
                # docstring end
                if lineno == lastlineindex:
                    mo = re.search(r"\S", linecont)
                    if (
                        mo is not None
                        and cb_info['cb_cur_indent'] <= mo.start()
                    ):
                        cb_info['cb_cur'].append(linecont)
                    if len(cb_info['cb_cur']):
                        _append_code_block(lineno > example_lineno)
                    break
                # check indent for cur block start and end.
                if cb_info['cb_cur_indent'] < 0:
                    mo = re.search(r"\S", linecont)
                    if mo is None:
                        continue
                    # find the first non empty line
                    cb_info['cb_cur_indent'] = mo.start()
                    cb_info['cb_cur'].append(linecont)
                else:
                    mo = re.search(r"\S", linecont)
                    if mo is None:
                        cb_info['cb_cur'].append(linecont)
                        continue
                    if cb_info['cb_cur_indent'] <= mo.start():
                        cb_info['cb_cur'].append(linecont)
                    else:
                        if linecont[mo.start()] == '#':
                            continue
                        else:
                            # block end
                            if len(cb_info['cb_cur']):
                                _append_code_block(lineno > example_lineno)
                            cb_info['cb_started'] = False
                            cb_info['cb_cur_indent'] = -1
                            cb_info['cb_cur'] = []
    return code_blocks


def find_last_future_line_end(cbstr):
    pat = re.compile('__future__.*\n')
    lastmo = None
    it = re.finditer(pat, cbstr)
    while True:
        try:
            lastmo = next(it)
        except StopIteration:
            break
    if lastmo:
        return lastmo.end()
    else:
        return None


def extract_sample_codes_into_dir():
    if os.path.exists(SAMPLECODE_TEMPDIR):
        if not os.path.isdir(SAMPLECODE_TEMPDIR):
            os.remove(SAMPLECODE_TEMPDIR)
            os.mkdir(SAMPLECODE_TEMPDIR)
    else:
        os.mkdir(SAMPLECODE_TEMPDIR)
    for id_api in api_info_dict:
        if (
            'docstring' in api_info_dict[id_api]
            and 'full_name' in api_info_dict[id_api]
        ):
            code_blocks = extract_code_blocks_from_docstr(
                api_info_dict[id_api]['docstring']
            )
            for cb_info in code_blocks:
                fn = os.path.join(
                    SAMPLECODE_TEMPDIR,
                    '{}.sample-code-{}.py'.format(
                        api_info_dict[id_api]['full_name'], cb_info['id']
                    ),
                )
                requires = cb_info['required']
                if not is_required_match(requires, fn):
                    continue
                with open(fn, 'w') as f:
                    header = None
                    # TODO: xpu, distribted
                    if 'gpu' in requires:
                        header = 'import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "{}"\n\n'.format(
                            GPU_ID
                        )
                    else:
                        header = 'import os\nos.environ["CUDA_VISIBLE_DEVICES"] = ""\n\n'
                    cb = cb_info['codes']
                    last_future_line_end = find_last_future_line_end(cb)
                    if last_future_line_end:
                        f.write(cb[:last_future_line_end])
                        f.write(header)
                        f.write(cb[last_future_line_end:])
                    else:
                        f.write(header)
                        f.write(cb)
                    f.write(
                        '\nprint("{} sample code is executed successfully!")'.format(
                            fn
                        )
                    )


def is_required_match(requires, cbtitle=''):
    """
    search the required instruction in the code-block, and check it match the current running environment.

    environment values of equipped: cpu, gpu, xpu, distributed, skip
    the 'skip' is the special flag to skip the test, so is_required_match will return False directly.
    """
    if 'skip' in requires:
        logger.info('%s: skipped', cbtitle)
        return False

    if all([k in EQUIPPED_DEVICES for k in requires]):
        return True

    logger.info(
        '%s: the equipments [%s] not match the required [%s].',
        cbtitle,
        ','.join(EQUIPPED_DEVICES),
        ','.join(requires),
    )
    return False


def get_requires_of_code_block(cbstr):
    requires = set(['cpu'])
    pat = re.compile(r'#\s*require[s|d]\s*:\s*(.*)')
    mo = re.search(pat, cbstr)
    if mo is None:
        # treat is as required: cpu
        return requires

    for r in mo.group(1).split(','):
        rr = r.strip().lower()
        if rr:
            requires.add(rr)
    return requires


def get_all_equippted_devices():
    ENV_KEY = 'TEST_ENVIRONMENT_EQUIPEMNT'
    if ENV_KEY in os.environ:
        for r in os.environ[ENV_KEY].split(','):
            rr = r.strip().lower()
            if r:
                EQUIPPED_DEVICES.add(rr)
    if 'cpu' not in EQUIPPED_DEVICES:
        EQUIPPED_DEVICES.add('cpu')

    EQUIPPED_DEVICES.add(RUN_ON_DEVICE)


def run_a_sample_code(sc_filename):
    succ = True
    cmd = None
    retstr = None
    if platform.python_version()[0] == "2":
        cmd = ["python", sc_filename]
    elif platform.python_version()[0] == "3":
        cmd = ["python3", sc_filename]
    else:
        retstr = 'Error: fail to parse python version!'
        logger.warning(retstr)
        succ = False
    subprc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, error = subprc.communicate()
    msg = "".join(output.decode(encoding='utf-8'))
    err = "".join(error.decode(encoding='utf-8'))
    if subprc.returncode != 0:
        retstr = """{} return code number {}.
stderr:
{}
stdout:
{}
""".format(
            sc_filename, subprc.returncode, err, msg
        )
        succ = False
    return succ, retstr


def run_all_sample_codes(threads=1):
    po = multiprocessing.Pool(threads)
    sc_files = os.listdir(SAMPLECODE_TEMPDIR)
    logger.info('there are %d sample codes to run', len(sc_files))
    mpresults = po.map_async(
        run_a_sample_code,
        [os.path.join(SAMPLECODE_TEMPDIR, fn) for fn in sc_files],
    )
    po.close()
    po.join()
    results = mpresults.get()

    err_files = []
    for i, fn in enumerate(sc_files):
        if not results[i][0]:
            err_files.append(fn)
            logger.warning(results[i][1])
    if len(err_files):
        logger.info(
            'there are %d sample codes run error.\n%s',
            len(err_files),
            "\n".join(err_files),
        )
        return False
    else:
        logger.info('all sample codes run successfully')
    return True


def reset_api_info_dict():
    global api_info_dict, parsed_mods
    api_info_dict = {}
    parsed_mods = {}


arguments = [
    # flags, dest, type, default, help
    ['--logf', 'logf', str, None, 'file for logging'],
    [
        '--attr',
        'travelled_attr',
        str,
        'all,dict',
        'the attribute for travelling, must be subset of [all,dict], such as "all" or "dict" or "all,dict".',
    ],
    [
        '--gen-rst',
        'gen_rst',
        bool,
        True,
        'generate English api reST files. If "all" in attr, only for "all".',
    ],
    [
        '--extract-sample-codes-dir',
        'sample_codes_dir',
        str,
        None,
        'if setted, the sample-codes will be extracted into the dir. If "all" in attr, only for "all".',
    ],
    ['--gpu_id', 'gpu_id', int, 0, 'GPU device id to use [0]'],
    ['--run-on-device', 'run_on_device', str, 'cpu', 'run on device'],
    [
        '--threads',
        'threads',
        int,
        1,
        'number of subprocesseses for running the all sample codes.',
    ],
]


def parse_args():
    """
    Parse input arguments
    """
    global arguments
    parser = argparse.ArgumentParser(
        description='generate the api_info json and generate the English api_doc reST files.'
    )
    parser.add_argument('--debug', dest='debug', action="store_true")
    parser.add_argument(
        '--run-sample-codes',
        dest='run_sample_codes',
        action="store_true",
        help='run all the smaple codes',
    )
    for item in arguments:
        parser.add_argument(
            item[0], dest=item[1], help=item[4], type=item[2], default=item[3]
        )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.logf:
        logfHandler = logging.FileHandler(args.logf)
        logfHandler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(logfHandler)
    if args.run_on_device.lower() == 'gpu':
        RUN_ON_DEVICE = 'gpu'
        GPU_ID = args.gpu_id

    get_all_equippted_devices()
    need_run_sample_codes = False
    if args.run_sample_codes or (
        'RUN_SAMPLE_CODES' in os.environ
        and os.environ['RUN_SAMPLE_CODES'].lower() in ['yes', '1', 'on']
    ):
        need_run_sample_codes = True
    if args.threads == 1 and (
        'RUN_SAMPLE_CODES_THREADS' in os.environ
        and int(os.environ['RUN_SAMPLE_CODES_THREADS']) > 1
    ):
        args.threads = int(os.environ['RUN_SAMPLE_CODES_THREADS'])
    if args.sample_codes_dir:
        SAMPLECODE_TEMPDIR = args.sample_codes_dir

    if 'VERSIONSTR' in os.environ and os.environ['VERSIONSTR'] == '1.8':
        # 1.8 not used
        docdirs = ['../beginners_guide', '../advanced_guide', '../user_guides']
    else:
        docdirs = ['../guides', '../tutorial']
    collect_referenced_from_infos(docdirs)

    realattrs = []  # only __all__ or __dict__
    for attr in args.travelled_attr.split(','):
        realattr = attr.strip()
        if realattr in ['all', '__all__']:
            realattr = '__all__'
        elif realattr in ['dict', '__dict__']:
            realattr = '__dict__'
        else:
            logger.warning("unknown value in attr: %s", attr)
            continue
        realattrs.append(realattr)
    for realattr in realattrs:
        jsonfn = None
        if realattr == '__all__':
            jsonfn = 'api_info_all.json'
        elif realattr == '__dict__':
            jsonfn = 'api_info_dict.json'
        else:
            continue

        logger.info("travelling attr: %s", realattr)
        reset_api_info_dict()
        get_all_api(attr=realattr)
        set_display_attr_of_apis()
        set_source_code_attrs()
        set_referenced_from_attr()
        set_api_sketch()
        insert_suggested_names()
        if ('__all__' not in realattrs) or (
            '__all__' in realattrs and realattr == '__all__'
        ):
            if args.gen_rst:
                gen_en_files()
                check_cn_en_match()
            if need_run_sample_codes:
                extract_sample_codes_into_dir()
        filter_out_object_of_api_info_dict()
        json.dump(api_info_dict, open(jsonfn, "w"), indent=4)

    if need_run_sample_codes:
        for package in ['scipy', 'paddle2onnx']:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package]
            )
        run_all_sample_codes(args.threads)

    logger.info("done")
