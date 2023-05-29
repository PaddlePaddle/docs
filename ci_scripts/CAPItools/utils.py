# 获取存在 PADDLE_API func 数组的名称
def get_PADDLE_API_func(data: dict):
    result = []
    for i in data["functions"]:
        if 'PADDLE_API' in i['debug']:
            result.append(i)
    return result


# 获取存在 PADDLE_API class 数组的名称
def get_PADDLE_API_class(data: dict):
    result = []
    for classname in data["classes"]:
        # TODO 目前没有 PADDLE_API 是 struct 的
        if data["classes"][classname]["declaration_method"] == "struct":
            continue

        # TODO 这里需要处理一下, 因为类名和 PADDLE_API 会粘在一起, 例: PADDLE_APIDeviceContextPool
        if "PADDLE_API" in classname:
            result.append(data["classes"][classname])
    return result


# 获取方法中的参数parameters
def get_parameters(parameters):
    # parameter_api = ""  # 这里解析是给api使用的 (暂时不用)
    parameter_dict = {}
    for i in parameters:
        parameter_type_tmp = i['type'].replace(" &", "").replace(" *", "")
        # * 和 & 情况
        # parameter_api += parameter_type_tmp
        if i["reference"] == 1:
            # parameter_api += "&"
            parameter_type_tmp += "&"
        if i["pointer"] == 1:
            # parameter_api += "*"
            parameter_type_tmp += "*"
        if i["constant"] == 1 and not parameter_type_tmp.startswith('const'):
            parameter_type_tmp = "const " + parameter_type_tmp
        # parameter_api += f" {i['name']}, "
        desc = i.get('desc', '').replace('  ', '')

        # special progress for none parameter name case
        if i['name'] == '&':
            continue
        else:
            parameter_dict[i['name']] = {
                'type': parameter_type_tmp,
                'intro': desc,
            }
        # parameter += f"\t- **{i['name']}** ({parameter_type_tmp}) - {desc}\n"
    # 去掉末尾的逗号
    # parameter_api = parameter_api[:-2]
    # return parameter, parameter_api
    return parameter_dict


def parse_doxygen(doxygen):
    doxygen_dict = {
        'intro': '',
        'returns': '',
        'param_intro': {},
        'note': '',
    }

    if '@' in doxygen:
        doxygen = doxygen[doxygen.find('@') :]
        for doxygen_part in doxygen.split('@'):
            if doxygen_part.startswith('brief '):
                doxygen_dict['intro'] = doxygen_part.replace('brief ', '', 1)
            elif doxygen_part.startswith('return '):
                doxygen_dict['returns'] = doxygen_part.replace('return ', '', 1)
            elif doxygen_part.startswith('param '):
                param_intro = doxygen_part.replace('param ', '', 1)
                param_name = param_intro[: param_intro.find(' ')]
                doxygen_dict['param_intro'][param_name] = param_intro[
                    param_intro.find(' ') + 1 :
                ]
            elif doxygen_part.startswith('note '):
                doxygen_dict['note'] = doxygen_part.replace('note ', '', 1)
            else:
                pass

    return doxygen_dict
