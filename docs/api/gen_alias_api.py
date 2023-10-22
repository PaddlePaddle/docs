import paddle
import inspect
import pkgutil
import os
import sys
import subprocess


class AliasAPIGen:
    def __init__(self, paddle_root_path):
        self.api_dict = {}
        self.root_module = paddle
        self.not_display_prefix = set(
            ["paddle.incubate", "paddle.fluid.contrib"]
        )
        self.id_api_dict = {}
        self.paddle_root_path = paddle_root_path

    def get_func_and_class_from_paddle(self):
        for n, obj in inspect.getmembers(self.root_module):
            if inspect.isclass(obj) or inspect.isfunction(obj):
                if obj.__name__.startswith("_"):
                    continue
                self.api_dict["paddle" + "." + obj.__name__] = id(obj)

        for filefiner, name, ispkg in pkgutil.walk_packages(
            path=self.root_module.__path__,
            prefix=self.root_module.__name__ + ".",
        ):
            try:
                m = eval(name)
            except AttributeError:
                pass
            else:
                for n, obj in inspect.getmembers(eval(name)):
                    if inspect.isclass(obj) or inspect.isfunction(obj):
                        if obj.__name__.startswith("_"):
                            continue
                        self.api_dict[name + "." + obj.__name__] = id(obj)

    def group_api_by_id(self):
        for key in self.api_dict:
            api_id = self.api_dict[key]
            if api_id in self.id_api_dict:
                self.id_api_dict[api_id].append(key)
            else:
                self.id_api_dict[api_id] = [key]

    def _choose_real_api(self, api_list):
        api = self._find_real_api_by_grep_file(api_list)
        if api != "":
            return api

        # find the apis with longest path,
        # if more than one, try to find the api with paddle.fluid
        max_len = 0
        max_len_apis = []
        for api in api_list:
            l = len(api.split("."))
            if l > max_len:
                max_len = l
                max_len_apis = [api]
            elif l == max_len:
                max_len_apis.append(api)

        for api in max_len_apis:
            if api.startswith("paddle.fluid"):
                return api
        return max_len_apis[0]

    # try to get the realization of the api by grep "def api_name" or "class api_name"
    def _find_real_api_by_grep_file(self, api_list):
        api_ok = ""
        for api in api_list:
            try:
                eval(api)
            except AttributeError:
                pass
            else:
                api_ok = api

        # can not find an api which is exist in the paddle
        if api_ok == "":
            return ""

        api = api_ok
        reg = ""
        api_last_name = api.split(".")[-1]

        obj = eval(api)
        if inspect.isclass(obj):
            reg = "class %s(" % api.split(".")[-1]
        elif inspect.isfunction(obj):
            reg = "def %s(" % api.split(".")[-1]

        shell_cmd = "find %s -name '*.py' | xargs grep  \"%s\" " % (
            self.paddle_root_path,
            reg,
        )

        p = subprocess.Popen(
            shell_cmd, shell=True, stdout=subprocess.PIPE, stderr=None
        )
        retval = p.wait()
        if retval == 0:
            result = list(p.stdout.readlines())
            for fp in result:
                grep_api = self.transform_file_to_api(fp, api_last_name)
                try:
                    grep_api_obj = eval(grep_api)
                except AttributeError:
                    pass
                else:
                    if id(grep_api_obj) == id(obj):
                        return grep_api

        return ""

    def transform_file_to_api(self, file_path, api_last_name):
        prefix = self.paddle_root_path

        fp = file_path.decode().split(".py")[0]
        tmp = fp.replace(prefix, "")
        tmp = tmp.replace("/", ".")
        api = tmp + "." + api_last_name
        return api

    # find the shortest path api which not starts with paddle.fluid
    def _choose_recomment_api(self, api_list):
        min_len = len(api_list[0].split("."))
        rec_api = api_list[0]
        for api in api_list:
            if not api.startswith("paddle.fluid"):
                if min_len > len(api.split(".")):
                    min_len = len(api.split("."))
                    rec_api = api
        return rec_api

    def sort_alias_name(self, api_list):
        real_api = self._choose_real_api(api_list)
        api_list.remove(real_api)

        rec_api = self._choose_recomment_api(api_list)
        api_list.remove(rec_api)

        # sort others api by path length
        api_list.sort(key=lambda api: api.count("."))

        return [real_api] + [rec_api] + api_list

    def filter_api(self, api_list):
        for api in api_list:
            for api_prefix in self.not_display_prefix:
                if api.startswith(api_prefix):
                    return True
        return False

    def format_print(self, api_list):
        print(api_list[0] + "\t" + ",".join(api_list[1:]))

    def generator_alias_api(self):
        self.get_func_and_class_from_paddle()
        self.group_api_by_id()

        for key in self.id_api_dict:
            if len(self.id_api_dict[key]) > 1:
                sorted_list = self.sort_alias_name(self.id_api_dict[key])
                if not self.filter_api(sorted_list):
                    self.format_print(sorted_list)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Useage:")
        print("python3 gen_alias_api.py path-to-paddle-root")
        exit(1)
    else:
        paddle_root = sys.argv[1]
        alias_gen = AliasAPIGen(paddle_root + "/python/")
        alias_gen.generator_alias_api()
