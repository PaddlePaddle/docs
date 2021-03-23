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
"""
generate api_info_dict.json to describe all info about the apis.
"""

en_suffix = "_en.rst"
cn_suffix = "_cn.rst"
NOT_DISPLAY_DOC_LIST_FILENAME = "./not_display_doc_list"
DISPLAY_DOC_LIST_FILENAME = "./display_doc_list"
ALIAS_MAPPING_LIST_FILENAME = "./alias_api_mapping"
SAMPLECODE_TEMPDIR = './sample-codes'
RUN_ON_DEVICE = "cpu"
GPU_ID = 0

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
parsed_mods = {}

logger = logging.getLogger()
if logger.handlers:
    # we assume the first handler is the one we want to configure
    console = logger.handlers[0]
else:
    console = logging.StreamHandler()
    logger.addHandler(console)
console.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"))


# step 1: walkthrough the paddle package to collect all the apis in api_set
def get_all_api(root_path='paddle', attr="__all__"):
    """
    walk through the paddle package to collect all the apis.
    """
    global api_info_dict
    api_counter = 0
    for filefinder, name, ispkg in pkgutil.walk_packages(
            path=paddle.__path__, prefix=paddle.__name__ + '.'):
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
    logger.info('%s: collected %d apis, %d distinct apis.', attr, api_counter,
                len(api_info_dict))


# step 1 fill field : `id` & `all_names`, type, docstring
def process_module(m, attr="__all__"):
    api_counter = 0
    if hasattr(m, attr):
        # may have duplication of api
        for api in set(getattr(m, attr)):
            if api[0] == '_': continue
            # Exception occurred when `id(eval(paddle.dataset.conll05.test, get_dict))`
            if ',' in api: continue

            # api's fullname
            full_name = m.__name__ + "." + api
            try:
                obj = eval(full_name)
                fc_id = id(obj)
            except AttributeError:
                logger.warning("AttributeError occurred when `id(eval(%s))`",
                               full_name)
                pass
            except:
                logger.warning("Exception occurred when `id(eval(%s))`",
                               full_name)
            else:
                api_counter += 1
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
                        api_info_dict[fc_id]["docstring"] = inspect.cleandoc(
                            docstr)
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
        logger.debug("processing %s:%s:%s", obj_type_name, item["id"],
                     str(obj))
        if obj_type_name == "module":
            if hasattr(obj, '__file__') and obj.__file__ is not None and len(
                    obj.__file__) > src_file_start_ind:
                api_info_dict[id_api]["src_file"] = obj.__file__[
                    src_file_start_ind:]
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
                    logger.debug("{}, {}, {}".format(item["id"], item["type"],
                                                     item["all_names"]))
            else:
                found = False
                for name in item["all_names"]:
                    mod_name, short_name = split_name(name)
                    if mod_name in sys.modules:
                        mod = sys.modules[mod_name]
                        parse_module_file(mod)
                        found = True
                if not found:
                    logger.debug("{}, {}, {}".format(item["id"], item["type"],
                                                     item["all_names"]))


def split_name(name):
    try:
        r = name.rindex('.')
        return [name[:r], name[r + 1:]]
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
        logger.debug("parsing %s:%s", mod_name, src_file)
        if len(mod_name) >= 6 and mod_name[:6] == 'paddle':
            if os.path.splitext(src_file)[1].lower() == '.py':
                mod_ast = ast.parse(open(src_file, "r").read())
                for node in mod_ast.body:
                    short_names = []
                    if ((isinstance(node, ast.ClassDef) or
                         isinstance(node, ast.FunctionDef)) and
                            hasattr(node, 'name') and
                            hasattr(sys.modules[mod_name],
                                    node.name) and node.name[0] != '_'):
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
                            logger.warning("%s maybe %s.%s", obj_full_name,
                                           mod.__package__, short_name)
                            obj_full_name = mod.__package__ + '.' + short_name
                            try:
                                obj_this = eval(obj_full_name)
                                obj_id = id(obj_this)
                            except:
                                continue
                        if obj_id in api_info_dict and "lineno" not in api_info_dict[
                                obj_id]:
                            api_info_dict[obj_id]["src_file"] = src_file[
                                src_file_start_ind:]
                            api_info_dict[obj_id][
                                "doc_filename"] = obj_full_name.replace('.',
                                                                        '/')
                            api_info_dict[obj_id]["full_name"] = obj_full_name
                            api_info_dict[obj_id]["short_name"] = short_name
                            api_info_dict[obj_id]["module_name"] = mod_name
                            api_info_dict[obj_id]["lineno"] = node.lineno
                            if has_end_lineno:
                                api_info_dict[obj_id][
                                    "end_lineno"] = node.end_lineno
                            if isinstance(node, ast.FunctionDef):
                                api_info_dict[obj_id][
                                    "args"] = gen_functions_args_str(node)
                            elif isinstance(node, ast.ClassDef):
                                for n in node.body:
                                    if hasattr(
                                            n,
                                            'name') and n.name == '__init__':
                                        api_info_dict[obj_id][
                                            "args"] = gen_functions_args_str(n)
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
                        if obj_id in api_info_dict and "lineno" not in api_info_dict[
                                obj_id]:
                            api_info_dict[obj_id]["src_file"] = src_file[
                                src_file_start_ind:]
                            api_info_dict[obj_id]["full_name"] = obj_full_name
                            api_info_dict[obj_id]["short_name"] = short_name
                            api_info_dict[obj_id]["module_name"] = mod_name
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
                    node.args.defaults[defarg_ind].id)
            elif isinstance(node.args.defaults[defarg_ind], ast.Constant):
                str_args_list[defarg_ind_start + defarg_ind] += '=' + str(
                    node.args.defaults[defarg_ind].value)
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
        display_none_apis = set([
            line.strip() for line in open(NOT_DISPLAY_DOC_LIST_FILENAME, "r")
        ])
    else:
        logger.warning("file not exists: %s", NOT_DISPLAY_DOC_LIST_FILENAME)
        display_none_apis = set()
    if os.path.exists(DISPLAY_DOC_LIST_FILENAME):
        display_yes_apis = set(
            [line.strip() for line in open(DISPLAY_DOC_LIST_FILENAME, "r")])
    else:
        logger.warning("file not exists: %s", DISPLAY_DOC_LIST_FILENAME)
        display_yes_apis = set()
    logger.info(
        'display_none_apis has %d items, display_yes_apis has %d items',
        len(display_none_apis), len(display_yes_apis))

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


# step 4 fill field : alias_name
def set_real_api_alias_attr():
    """
    set the full_name,alias attr and so on.
    """
    if not os.path.exists(ALIAS_MAPPING_LIST_FILENAME):
        logger.warning("file not exists: %s", ALIAS_MAPPING_LIST_FILENAME)
        return
    for line in open(ALIAS_MAPPING_LIST_FILENAME, "r"):
        linecont = line.strip()
        lineparts = linecont.split()
        if len(lineparts) < 2:
            logger.warning('line "{}" splited to {}'.format(line, lineparts))
            continue
        try:
            real_api = lineparts[0].strip()
            m = eval(real_api)
        except AttributeError:
            logger.warning("AttributeError: %s", real_api)
        else:
            api_id = id(m)
            if api_id in api_info_dict:
                api_info_dict[api_id]["alias_name"] = lineparts[1]
                docpath_from_real_api = real_api.replace('.', '/')
                if "doc_filename" not in api_info_dict[api_id]:
                    api_info_dict[api_id][
                        "doc_filename"] = docpath_from_real_api
                else:
                    if api_info_dict[api_id][
                            "doc_filename"] != docpath_from_real_api:
                        logger.warning("doc_filename changes from %s to %s",
                                       api_info_dict[api_id]["doc_filename"],
                                       docpath_from_real_api)
                        api_info_dict[api_id][
                            "doc_filename"] = docpath_from_real_api
                if "module_name" not in api_info_dict[
                        api_id] or "short_name" not in api_info_dict[api_id]:
                    mod_name, short_name = split_name(real_api)
                    api_info_dict[api_id]["module_name"] = mod_name
                    api_info_dict[api_id]["short_name"] = short_name
                    if 'full_name' not in api_info_dict[api_id]:
                        api_info_dict[api_id]["full_name"] = real_api


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


# using `doc_filename`
def gen_en_files(api_label_file="api_label"):
    """
    generate all the en doc files.
    """
    with open(api_label_file, 'w') as api_label:
        for id_api, api_info in api_info_dict.items():
            # api_info = api_info_dict[id_api]
            if "display" in api_info and not api_info["display"]:
                logger.debug("{} display False".format(id_api))
                continue
            if "doc_filename" not in api_info:
                logger.debug(
                    "{} does not have doc_filename field.".format(id_api))
                continue
            else:
                logger.debug(api_info["doc_filename"])
            path = os.path.dirname(api_info["doc_filename"])
            if not os.path.exists(path):
                os.makedirs(path)
            f = api_info["doc_filename"] + en_suffix
            if os.path.exists(f):
                continue
            gen = EnDocGenerator()
            with gen.guard(f):
                if 'full_name' in api_info:
                    mod_name, _, short_name = api_info['full_name'].rpartition(
                        '.')
                else:
                    mod_name = api_info['module_name']
                    short_name = api_info['short_name']
                    logger.warning("full_name not in api_info: %s.%s",
                                   mod_name, short_name)
                gen.module_name = mod_name
                gen.api = short_name
                gen.print_header_reminder()
                gen.print_item()
                api_label.write("{1}\t.. _api_{0}_{1}:\n".format("_".join(
                    mod_name.split(".")), short_name))


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
            logger.warning("attribute error: module_name=" + self.module_name +
                           ", api=" + self.api)
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


def filter_api_info_dict():
    for id_api in api_info_dict:
        if "all_names" in api_info_dict[id_api]:
            api_info_dict[id_api]["all_names"] = list(
                api_info_dict[id_api]["all_names"])
        if "object" in api_info_dict[id_api]:
            del api_info_dict[id_api]["object"]


def extract_code_blocks_from_docstr(docstr):
    """
    extract code-blocks from the given docstring.

    DON'T include the multiline-string definition in code-blocks.

    Args:
        docstr - docstring
    Return:
        A list of code-blocks, indent removed.
    """
    code_blocks = []
    mo = re.search(r"Examples:", docstr)
    if mo is None:
        return code_blocks
    ds_list = docstr[mo.start():].replace("\t", '    ').split("\n")
    lastlineindex = len(ds_list) - 1
    cb_started = False
    cb_start_pat = re.compile(r"code-block::\s*python")
    cb_cur = []
    cb_cur_indent = -1
    for lineno, linecont in enumerate(ds_list):
        if re.search(cb_start_pat, linecont):
            if not cb_started:
                cb_started = True
                continue
            else:
                # cur block end
                if len(cb_cur):
                    code_blocks.append(inspect.cleandoc("\n".join(cb_cur)))
                cb_started = True  # another block started
                cb_cur_indent = -1
                cb_cur = []
        else:
            # check indent for cur block ends.
            if cb_started:
                if lineno == lastlineindex:
                    mo = re.search(r"\S", linecont)
                    if mo is not None and cb_cur_indent <= mo.start():
                        cb_cur.append(linecont)
                    if len(cb_cur):
                        code_blocks.append(inspect.cleandoc("\n".join(cb_cur)))
                    break
                if cb_cur_indent < 0:
                    mo = re.search(r"\S", linecont)
                    if mo is None: continue
                    cb_cur_indent = mo.start()
                    cb_cur.append(linecont)
                else:
                    mo = re.search(r"\S", linecont)
                    if mo is None: continue
                    if cb_cur_indent <= mo.start():
                        cb_cur.append(linecont)
                    else:
                        if linecont[mo.start()] == '#':
                            continue
                        else:
                            # block end
                            if len(cb_cur):
                                code_blocks.append(
                                    inspect.cleandoc("\n".join(cb_cur)))
                            cb_started = False
                            cb_cur_indent = -1
                            cb_cur = []
    return code_blocks


def extract_sample_codes_into_dir():
    if os.path.exists(SAMPLECODE_TEMPDIR):
        if not os.path.isdir(SAMPLECODE_TEMPDIR):
            os.remove(SAMPLECODE_TEMPDIR)
            os.mkdir(SAMPLECODE_TEMPDIR)
    else:
        os.mkdir(SAMPLECODE_TEMPDIR)
    header = None
    if RUN_ON_DEVICE == "cpu":
        header = 'import os\nos.environ["CUDA_VISIBLE_DEVICES"] = ""\n\n'
    elif RUN_ON_DEVICE == "gpu":
        header = 'import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "{}"\n\n'.format(
            GPU_ID)
    for id_api in api_info_dict:
        if 'docstring' in api_info_dict[
                id_api] and 'full_name' in api_info_dict[id_api]:
            code_blocks = extract_code_blocks_from_docstr(
                api_info_dict[id_api]['docstring'])
            for cb_ind, cb in enumerate(code_blocks):
                fn = os.path.join(
                    SAMPLECODE_TEMPDIR, '{}.sample-code-{}.py'.format(
                        api_info_dict[id_api]['full_name'], cb_ind))
                with open(fn, 'w') as f:
                    f.write(header)
                    f.write(cb)
                    f.write(
                        '\nprint("{} sample code is executed successfully!")'.
                        format(fn))


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
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = subprc.communicate()
    msg = "".join(output.decode(encoding='utf-8'))
    err = "".join(error.decode(encoding='utf-8'))
    if subprc.returncode != 0:
        retstr = """{} return code number {}.
stderr:
{}
stdout:
{}
""".format(sc_filename, subprc.returncode, err, msg)
        succ = False
    return succ, retstr


def run_all_sample_codes(threads=1):
    po = multiprocessing.Pool(threads)
    sc_files = os.listdir(SAMPLECODE_TEMPDIR)
    logger.info('there are %d sample codes to run', len(sc_files))
    mpresults = po.map_async(
        run_a_sample_code,
        [os.path.join(SAMPLECODE_TEMPDIR, fn) for fn in sc_files])
    po.close()
    po.join()
    results = mpresults.get()

    err_files = []
    for i, fn in enumerate(sc_files):
        if not results[i][0]:
            err_files.append(fn)
            logger.warning(results[i][1])
    if len(err_files):
        logger.info('there are %d samplecodes run error.\n%s',
                    len(err_files), "\n".join(err_files))
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
        '--attr', 'travelled_attr', str, 'all,dict',
        'the attribute for travelling, must be subset of [all,dict], such as "all" or "dict" or "all,dict".'
    ],
    [
        '--gen-rst', 'gen_rst', bool, True,
        'genrate English api_cod reST files. If "all" in attr, only for "all".'
    ],
    [
        '--extract-sample-codes-dir', 'sample_codes_dir', str, None,
        'if setted, the sample-codes will be extracted into the dir. If "all" in attr, only for "all".'
    ],
    ['--gpu_id', 'gpu_id', int, 0, 'GPU device id to use [0]'],
    ['--run-on-device', 'run_on_device', str, 'cpu', 'run on device'],
    [
        '--threads', 'threads', int, 1,
        'subprocesses\' number for running the all sample codes.'
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
        help='run all the smaple codes')
    for item in arguments:
        parser.add_argument(
            item[0], dest=item[1], help=item[4], type=item[2], default=item[3])

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
            ))
        logger.addHandler(logfHandler)
    if args.run_on_device.lower() == 'gpu':
        RUN_ON_DEVICE = 'gpu'
        GPU_ID = args.gpu_id
    if args.run_sample_codes or (
            'RUN_SAMPLE_CODES' in os.environ and
            os.environ['RUN_SAMPLE_CODES'].lower() in ['yes', '1', 'on']):
        need_run_sample_codes = True
    if args.threads == 1 and ('RUN_SAMPLE_CODES_THREADS' in os.environ and
                              int(os.environ['RUN_SAMPLE_CODES_THREADS']) > 1):
        args.threads = int(os.environ['RUN_SAMPLE_CODES_THREADS'])
    if args.sample_codes_dir:
        SAMPLECODE_TEMPDIR = args.sample_codes_dir

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
        set_real_api_alias_attr()
        filter_api_info_dict()
        json.dump(api_info_dict, open(jsonfn, "w"), indent=4)
        if ('__all__' not in realattrs) or ('__all__' in realattrs and
                                            realattr == '__all__'):
            if args.gen_rst:
                gen_en_files()
                check_cn_en_match()
            if need_run_sample_codes:
                extract_sample_codes_into_dir()

    if need_run_sample_codes:
        run_all_sample_codes(args.threads)
    logger.info("done")
