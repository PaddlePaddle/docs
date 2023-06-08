import os

from utils import get_parameters, parse_doxygen

# 用于生成API文档的辅助类
# __init__ 初始化函数，调用decode
# decode 用于解析CppHeaderParser的解析信息
# create_and_write_file 根据指定的语言类型，在指定目录生成对应的文档
class func_helper(object):
    def __init__(self, function_dict, cpp2py_api_list):
        super(func_helper, self).__init__()
        self.function_dict = function_dict
        self.cpp2py_api_list = cpp2py_api_list
        self.decode()

    def decode(self):
        # 解析 api 信息
        self.func_name = self.function_dict["name"]
        self.api = self.function_dict["debug"].replace("PADDLE_API ", "")
        self.namespace = self.function_dict["namespace"].replace("::", "_")
        doxygen = (
            self.function_dict.get("doxygen", "")
            .replace("/**", "")
            .replace("*/", "")
            .replace("\n*", "")
            .replace("  ", "")
        )
        self.introduction = doxygen

        self.note = ""

        self.file_path = self.function_dict["filename"].replace("../", "")

        if len(self.function_dict["parameters"]) != 0:
            self.parameter_dict = get_parameters(
                self.function_dict["parameters"]
            )
        else:
            self.parameter_dict = {}

        self.returns = self.function_dict["returns"].replace("PADDLE_API ", "")

        # analysis doxygen
        doxygen_dict = parse_doxygen(doxygen)
        if doxygen_dict['intro'] != "":
            self.introduction = doxygen_dict['intro']
        if doxygen_dict['note'] != "":
            self.note = doxygen_dict['note']
        if doxygen_dict['returns'] != "":
            self.returns = doxygen_dict['returns']
        if doxygen_dict['param_intro'] != {}:
            for param_name in doxygen_dict['param_intro'].keys():
                self.parameter_dict[param_name]['intro'] = doxygen_dict[
                    'param_intro'
                ][param_name]

    def create_and_write_file(self, save_dir, language):
        if language == 'cn':
            self.create_and_write_file_cn(save_dir, language)
        elif language == 'en':
            self.create_and_write_file_en(save_dir, language)
        else:
            print('Error language! ')

    def create_and_write_file_cn(self, save_dir, language):
        with open(save_dir, 'w', encoding='utf8') as f:
            head_text = (
                f'.. _{language}_api_{self.namespace}{self.func_name}:\n' f'\n'
            )
            f.write(head_text)

            name_and_intro_text = (
                f'{self.func_name}\n'
                f'-------------------------------\n'
                f'\n'
                f'..cpp: function::{self.api}\n'
                f'{self.introduction}\n'
                f'\n'
            )
            f.write(name_and_intro_text)

            if self.func_name in self.cpp2py_api_list:
                cpp2py_text = (
                    f'本 API 与 Python API 对齐，详细用法可参考链接：'
                    f'[paddle.{self.func_name}]'
                    f'(https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/{self.func_name}_{language}.html)\n'
                    f'\n'
                )
                f.write(cpp2py_text)

            if self.note != "":
                note_text = f'..note::\n' f'\t{self.note}\n' f'\n'
                f.write(note_text)

            define_path_text = (
                f'定义目录\n' f':::::::::::::::::::::\n' f'{self.file_path}\n' f'\n'
            )
            f.write(define_path_text)

            if len(self.parameter_dict) != 0:
                parameters_text = f'参数\n' f':::::::::::::::::::::'
                f.write(parameters_text + '\n')
                for param in self.parameter_dict.keys():
                    param_text = f"\t- **{param}**"
                    if self.parameter_dict[param]['type'] != "":
                        param_text += f" ({self.parameter_dict[param]['type']})"
                    if self.parameter_dict[param]['intro'] != "":
                        param_text += (
                            f" - {self.parameter_dict[param]['intro']}"
                        )
                    param_text += "\n"
                    f.write(param_text)
            f.write('\n')

            return_text = (
                f'返回\n' f':::::::::::::::::::::\n' f'{self.returns}' f'\n'
            )
            if 'void' not in self.returns:
                f.write(return_text)

    def create_and_write_file_en(self, save_dir, language):
        with open(save_dir, 'w', encoding='utf8') as f:
            head_text = (
                f'.. _{language}_api_{self.namespace}{self.func_name}:\n' f'\n'
            )
            f.write(head_text)

            name_and_intro_text = (
                f'{self.func_name}\n'
                f'-------------------------------\n'
                f'\n'
                f'..cpp: function::{self.api}\n'
                f'{self.introduction}\n'
                f'\n'
            )
            f.write(name_and_intro_text)

            if self.func_name in self.cpp2py_api_list:
                cpp2py_text = (
                    f'This API is aligned with Python API, more details are shown in '
                    f'[paddle.{self.func_name}]'
                    f'(https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/{self.func_name}_{language}.html)\n'
                    f'\n'
                )
                f.write(cpp2py_text)

            if self.note != "":
                note_text = f'..note::\n' f'\t{self.note}\n' f'\n'
                f.write(note_text)

            define_path_text = (
                f'Path\n' f':::::::::::::::::::::\n' f'{self.file_path}\n' f'\n'
            )
            f.write(define_path_text)

            if len(self.parameter_dict) != 0:
                parameters_text = f'Parameters\n' f':::::::::::::::::::::'
                f.write(parameters_text + '\n')
                for param in self.parameter_dict.keys():
                    param_text = f"\t- **{param}**"
                    if self.parameter_dict[param]['type'] != "":
                        param_text += f" ({self.parameter_dict[param]['type']})"
                    if self.parameter_dict[param]['intro'] != "":
                        param_text += (
                            f" - {self.parameter_dict[param]['intro']}"
                        )
                    param_text += "\n"
                    f.write(param_text)
            f.write('\n')

            return_text = (
                f'Returns\n' f':::::::::::::::::::::\n' f'{self.returns}' f'\n'
            )
            if 'void' not in self.returns:
                f.write(return_text)


# 用于生成Class文档的辅助类
# __init__ 初始化函数，调用decode
# decode 用于解析CppHeaderParser的解析信息
# create_and_write_file 根据指定的语言类型，在指定目录生成对应的文档
class class_helper(object):
    def __init__(self, class_dict):
        super(class_helper, self).__init__()
        self.class_dict = class_dict
        self.decode()

    def decode(self):
        self.branch = "develop"  # Note 这里可以看看从包里面获取
        self.class_name = self.class_dict["name"].replace("PADDLE_API", "")
        self.file_path = self.class_dict["filename"].replace("../", "")
        doxygen = (
            self.class_dict.get("doxygen", "")
            .replace("/**", "")
            .replace("*/", "")
            .replace("\n*", "")
            .replace("  ", "")
        )
        self.introduction = doxygen
        self.note = ""
        # analysis doxygen
        doxygen_dict = parse_doxygen(doxygen)
        if doxygen_dict['intro'] != "":
            self.introduction = doxygen_dict['intro']
        if doxygen_dict['note'] != "":
            self.note = doxygen_dict['note']

        # 初始化函数
        # 避免空函数解析
        self.init_func = self.class_name

        self.functions_infor = []
        # Note: 未来可能在private也有函数
        # Note: 函数内构造函数可能解析有问题，需要后期查验
        self.class_function_number = len(self.class_dict["methods"]["public"])
        for i in range(self.class_function_number):
            ith_function = self.class_dict["methods"]["public"][i]

            function_name = ith_function['debug']
            # 获取描述
            funcs_doxygen = (
                ith_function.get("doxygen", "")
                .replace("/**", "")
                .replace("*/", "")
                .replace("\n*", "")
                .replace("  ", "")
            )
            funcs_intro = funcs_doxygen
            funcs_note = ""

            # 解析参数
            if len(ith_function["parameters"]) != 0:
                parameter_dict = get_parameters(ith_function["parameters"])
            else:
                parameter_dict = {}
            # 获取返回值
            # returns = ith_function["returns"].replace("PADDLE_API ", "")
            returns = ith_function["rtnType"]
            # Note Template 没有仅对class起作用，可能需要同步添加到API中
            template = ""
            if ith_function['template'] != False:
                template = ith_function['template']

            # analysis doxygen
            doxygen_dict = parse_doxygen(funcs_doxygen)
            if doxygen_dict['intro'] != "":
                funcs_intro = doxygen_dict['intro']
            if doxygen_dict['note'] != "":
                funcs_note = doxygen_dict['note']
            if doxygen_dict['returns'] != "":
                returns = doxygen_dict['returns']
            if doxygen_dict['param_intro'] != {}:
                for param_name in doxygen_dict['param_intro'].keys():
                    # Note: 可能param_name 不同步，需要注意
                    if param_name in parameter_dict.keys():
                        parameter_dict[param_name]['intro'] = doxygen_dict[
                            'param_intro'
                        ][param_name]

            self.functions_infor.append(
                {
                    'name': function_name,
                    'doxygen': funcs_intro,
                    'note': funcs_note,
                    'parameter': parameter_dict,
                    'returns': returns,
                    'template': template,
                }
            )

        # if '@' in self.doxygen:
        #     print('CLASS: ' + self.file_path + ' - ' + self.class_name)

    def create_and_write_file(self, save_dir, language):
        if language == 'cn':
            self.create_and_write_file_cn(save_dir, language)
        elif language == 'en':
            self.create_and_write_file_en(save_dir, language)
        else:
            print('Error language! ')

    def create_and_write_file_cn(self, save_dir, language):
        with open(save_dir, 'w', encoding='utf8') as f:
            head_text = f'.. _{language}_api_{self.class_name}:\n' f'\n'
            f.write(head_text)

            name_and_intro_text = (
                f'{self.class_name}[源代码](https://github.com/PaddlePaddle/Paddle/blob/{self.branch}/{self.file_path})\n'
                f'-------------------------------\n'
                f'\n'
                f'.. cpp:class:: {self.init_func}\n'
                f'{self.introduction}\n'
                f'\n'
            )
            f.write(name_and_intro_text)

            if self.note != "":
                note_text = f'..note::\n' f'\t{self.note}\n' f'\n'
                f.write(note_text)

            define_path_text = (
                f'定义目录\n' f':::::::::::::::::::::\n' f'{self.file_path}\n' f'\n'
            )
            f.write(define_path_text)

            if self.class_function_number != 0:
                class_function_head_text = (
                    f'方法\n' f':::::::::::::::::::::\n' f'\n'
                )
                f.write(class_function_head_text)

                for fun_infor in self.functions_infor:
                    if fun_infor['template'] == "":
                        fun_name_and_intro_text = ""
                    else:
                        fun_name_and_intro_text = f'{fun_infor["template"]}\n'
                    fun_name_and_intro_text += (
                        f"{fun_infor['name']}\n"
                        f"\'\'\'\'\'\'\'\'\'\'\'\n"
                        f"{fun_infor['doxygen']}\n"
                        f"\n"
                    )
                    f.write(fun_name_and_intro_text)

                    if fun_infor['note'] != "":
                        fun_note_text = (
                            f'..note::\n' f'\t{fun_infor["note"]}\n' f'\n'
                        )
                        f.write(fun_note_text)

                    if len(fun_infor['parameter']) != 0:
                        parameters_text = (
                            f"**参数**\n" f"\'\'\'\'\'\'\'\'\'\'\'\n"
                        )
                        f.write(parameters_text)
                        for param in fun_infor['parameter'].keys():
                            param_text = f"\t- **{param}**"
                            if fun_infor['parameter'][param]['type'] != "":
                                param_text += f" ({fun_infor['parameter'][param]['type']})"
                            if fun_infor['parameter'][param]['intro'] != "":
                                param_text += f" - {fun_infor['parameter'][param]['intro']}"
                            param_text += "\n"
                            f.write(param_text)
                    f.write('\n')

                    if (
                        fun_infor['returns'] != ''
                        and 'void' not in fun_infor['returns']
                    ):
                        fun_return_text = (
                            f"**返回**\n"
                            f"\'\'\'\'\'\'\'\'\'\'\'\n"
                            f"{fun_infor['returns']}\n"
                            f"\n"
                        )
                        f.write(fun_return_text)

    def create_and_write_file_en(self, save_dir, language):
        with open(save_dir, 'w', encoding='utf8') as f:
            head_text = f'.. _{language}_api_{self.class_name}:\n' f'\n'
            f.write(head_text)

            name_and_intro_text = (
                f'{self.class_name}[source](https://github.com/PaddlePaddle/Paddle/blob/{self.branch}/{self.file_path})\n'
                f'-------------------------------\n'
                f'\n'
                f'.. cpp:class:: {self.init_func}\n'
                f'{self.introduction}\n'
                f'\n'
            )
            f.write(name_and_intro_text)

            if self.note != "":
                note_text = f'..note::\n' f'\t{self.note}\n' f'\n'
                f.write(note_text)

            define_path_text = (
                f'Path\n' f':::::::::::::::::::::\n' f'{self.file_path}\n' f'\n'
            )
            f.write(define_path_text)

            if self.class_function_number != 0:
                class_function_head_text = (
                    f'Methods\n' f':::::::::::::::::::::\n' f'\n'
                )
                f.write(class_function_head_text)

                for fun_infor in self.functions_infor:
                    if fun_infor['template'] == "":
                        fun_name_and_intro_text = ""
                    else:
                        fun_name_and_intro_text = f'{fun_infor["template"]}\n'
                    fun_name_and_intro_text += (
                        f"{fun_infor['name']}\n"
                        f"\'\'\'\'\'\'\'\'\'\'\'\n"
                        f"{fun_infor['doxygen']}\n"
                        f"\n"
                    )
                    f.write(fun_name_and_intro_text)

                    if fun_infor['note'] != "":
                        fun_note_text = (
                            f'..note::\n' f'\t{fun_infor["note"]}\n' f'\n'
                        )
                        f.write(fun_note_text)

                    if len(fun_infor['parameter']) != 0:
                        parameters_text = (
                            f"**Parameters**\n" f"\'\'\'\'\'\'\'\'\'\'\'\n"
                        )
                        f.write(parameters_text)
                        for param in fun_infor['parameter'].keys():
                            param_text = f"\t- **{param}**"
                            if fun_infor['parameter'][param]['type'] != "":
                                param_text += f" ({fun_infor['parameter'][param]['type']})"
                            if fun_infor['parameter'][param]['intro'] != "":
                                param_text += f" - {fun_infor['parameter'][param]['intro']}"
                            param_text += "\n"
                            f.write(param_text)
                    f.write('\n')

                    if (
                        fun_infor['returns'] != ''
                        and 'void' not in fun_infor['returns']
                    ):
                        fun_return_text = (
                            f"**Returns**\n"
                            f"\'\'\'\'\'\'\'\'\'\'\'\n"
                            f"{fun_infor['returns']}\n"
                            f"\n"
                        )
                        f.write(fun_return_text)


# 用于生成Overview页面
# 根据指定的语言类型，在指定目录生成总览文档
def generate_overview(overview_list, save_dir, language):
    if language == 'cn':
        generate_overview_cn(overview_list, save_dir, language)
    elif language == 'en':
        generate_overview_en(overview_list, save_dir, language)
    else:
        print('Error language! ')


def generate_overview_cn(overview_list, root_dir, LANGUAGE):
    dir_path = os.path.join(root_dir, LANGUAGE)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    rst_dir = os.path.join(dir_path, 'index.rst')
    with open(rst_dir, 'w', encoding='utf8') as f:
        head_text = (
            f'# C++ 文档\n'
            f'欢迎使用飞桨框架（PaddlePaddle），PaddlePaddle 是一个易用、高效、灵活、可扩展的深度学习框架，致力于让深度学习技术的创新与应用更简单。\n'
            f'在本版本中，飞桨框架对 C++ 接口做了许多优化，您可以参考下表来了解飞桨框架最新版的 C++ 目录结构与说明。此外，您可参考 PaddlePaddle 的 GitHub 了解详情。\n'
            f'本文档的应用场景为 C++ 训练，并主要在自定义算子开发时使用。本文档内容持续迭代中，在下个版本可能会有不兼容的升级，如果不介意随下一版本升级的话，可以使用，追求稳定的话则不建议使用。\n'
            f'\n'
        )
        f.write(head_text)

        f.write('## 头文件索引\n')
        namespace_dict = {}  # 用于对齐namespace

        for h_dict in overview_list:
            basename = os.path.basename(h_dict["h_file"])
            h_head_text = f'### [{basename}]({h_dict["h_file"]})\n'
            f.write(h_head_text)

            # Note: add url link
            if len(h_dict["class"]) > 0:
                # write class
                h_class_text = f'#### classes\n'
                f.write(h_class_text)
                for class_name in h_dict["class"]:
                    class_namespace = class_name["namespace"] + "::"
                    # 在这里初始化字典为一个数组
                    if class_namespace not in namespace_dict.keys():
                        namespace_dict[class_namespace] = []
                    namespace_dict[class_name["namespace"] + "::"].append(
                        class_name['name'].replace("PADDLE_API", "")
                    )
                    f.write(
                        '- '
                        + class_name['name'].replace("PADDLE_API", "")
                        + '\n'
                    )

            if len(h_dict["function"]) > 0:
                # write functions
                h_function_text = f'#### functions\n'
                f.write(h_function_text)
                for function_name in h_dict["function"]:
                    if function_name["namespace"] not in namespace_dict.keys():
                        namespace_dict[function_name["namespace"]] = []
                    namespace_dict[function_name["namespace"]].append(
                        function_name['name']
                    )
                    f.write('- ' + function_name['name'] + '\n')

            f.write('\n')

        # 根据 namespace 进行分级写入
        namespace_text = '## 命名空间索引\n'
        for namespace in namespace_dict.keys():
            namespace_text += f'### {namespace}\n'
            for name in namespace_dict[namespace]:
                namespace_text += f'- {name}\n'
            namespace_text += '\n'
        f.write(namespace_text)


# 与 generate_overview_cn 实现原理一致
def generate_overview_en(overview_list, root_dir, LANGUAGE):
    dir_path = os.path.join(root_dir, LANGUAGE)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    rst_dir = os.path.join(dir_path, 'index.rst')
    with open(rst_dir, 'w', encoding='utf8') as f:
        head_text = (
            f'# C++ API Reference\n'
            f'PaddlePaddle (PArallel Distributed Deep LEarning) is an efficient, flexible, and extensible deep learning framework, commits to making the innovation and application of deep learning technology easier.\n'
            f'In this version, PaddlePaddle has made many optimizations to the C++ APIs. You can refer to the following table to understand the C++ API directory structure and description of the latest version of PaddlePaddle. In addition, you can refer to PaddlePaddle’s GitHub for details.\n'
            f'The application scenario of this document is C++training and is mainly used in the development of custom operators. The content of this document is continuously iterating, and there may be incompatible upgrades in the next version. If you don’t mind upgrading with the next version, you can use it. Otherwise, it is not recommended to use it.\n'
            f'\n'
        )
        f.write(head_text)

        f.write('## Index by header file\n')
        namespace_dict = {}

        for h_dict in overview_list:
            basename = os.path.basename(h_dict["h_file"])
            h_head_text = f'### [{basename}]({h_dict["h_file"]})\n'
            f.write(h_head_text)

            # Note: add url link
            if len(h_dict["class"]) > 0:
                # write class
                h_class_text = f'#### classes\n'
                f.write(h_class_text)
                for class_name in h_dict["class"]:
                    class_namespace = class_name["namespace"] + "::"
                    if class_namespace not in namespace_dict.keys():
                        namespace_dict[class_namespace] = []
                    namespace_dict[class_name["namespace"] + "::"].append(
                        class_name['name'].replace("PADDLE_API", "")
                    )
                    f.write(
                        '- '
                        + class_name['name'].replace("PADDLE_API", "")
                        + '\n'
                    )

            if len(h_dict["function"]) > 0:
                # write functions
                h_function_text = f'#### functions\n'
                f.write(h_function_text)
                for function_name in h_dict["function"]:
                    if function_name["namespace"] not in namespace_dict.keys():
                        namespace_dict[function_name["namespace"]] = []
                    namespace_dict[function_name["namespace"]].append(
                        function_name['name']
                    )
                    f.write('- ' + function_name['name'] + '\n')

            f.write('\n')

        namespace_text = '## Index by namespace\n'
        for namespace in namespace_dict.keys():
            namespace_text += f'### {namespace}\n'
            for name in namespace_dict[namespace]:
                namespace_text += f'- {name}\n'
            namespace_text += '\n'
        f.write(namespace_text)
