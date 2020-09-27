import paddle
import inspect
import pkgutil


class AliasAPIGen:
    def __init__(self):
        self.api_dict = {}
        self.root_module = paddle
        self.not_display_prefix = set(["paddle.incubate", "paddle.fluid.contrib"])
        self.id_api_dict = {}
    

    def get_func_and_class_from_paddle(self):
        for n, obj in inspect.getmembers(self.root_module):
            if inspect.isclass(obj) or inspect.isfunction(obj):
                if obj.__name__.startswith("_"):
                    continue
                self.api_dict["paddle" + "." + obj.__name__] = id(obj)

        for filefiner, name, ispkg in pkgutil.walk_packages(path=self.root_module.__path__,
            prefix=self.root_module.__name__ + "."):
            
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


    # find the apis with longest path,
    # if more than one, try to find the api with paddle.fluid
    def _choose_real_api(self, api_list):
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

        #sort others api by path length
        api_list.sort(key=lambda api: api.count("."))

        return [real_api] + [rec_api] + api_list


    def filter_api(self, api_list):
        for api in api_list:
            for api_prefix in self.not_display_prefix:
                if api.startswith(api_prefix):
                    return True
        return False


    def format_print(self, api_list):
        print(api_list[0] +  "\t" + ",".join(api_list[1:]))


    def generator_alias_api(self):
        self.get_func_and_class_from_paddle()
        self.group_api_by_id()

        for key in self.id_api_dict:
            if len(self.id_api_dict[key]) > 1:
                sorted_list = self.sort_alias_name(self.id_api_dict[key])
                if not self.filter_api(sorted_list):
                    self.format_print(sorted_list)


if __name__ == "__main__":
    alias_gen = AliasAPIGen()
    alias_gen.generator_alias_api()