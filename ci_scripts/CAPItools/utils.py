# 获取存在 PADDLE_API func 数组的名称
# CppHeaderParser 解析后以字典形式保存数据，'debug' 字段中保存了原始信息
# 如果 PADDLE_API 在字段中，则表明该 API 是外部暴露的函数
def get_PADDLE_API_func(data: dict):
    result = []
    for i in data["functions"]:
        if 'PADDLE_API' in i['debug']:
            result.append(i)
    return result


# 获取存在 PADDLE_API class 数组的名称
# CppHeaderParser 解析后以字典形式保存数据
# 如果 PADDLE_API 在字段中，则表明该 API 是外部暴露的类
def get_PADDLE_API_class(data: dict):
    result = []
    for classname in data["classes"]:
        # Note 目前没有 PADDLE_API 是 struct 的
        if data["classes"][classname]["declaration_method"] == "struct":
            continue

        # Note 这里需要处理一下, 因为类名和 PADDLE_API 会粘在一起, 例: PADDLE_APIDeviceContextPool
        if "PADDLE_API" in classname:
            result.append(data["classes"][classname])
    return result


# 获取方法中的参数parameters
# 根据解析的参数字典，添加对应的参数名、参数类型、说明
# 有时候会将“&”解析为参数名，需要特殊处理
def get_parameters(parameters):
    # parameter_api = ""  # 这里解析是给api使用的 (暂时不用)
    parameter_dict = {}
    for i in parameters:
        parameter_type_tmp = i['type'].replace(" &", "").replace(" *", "")
        # * 和 & 情况
        # parameter_api += parameter_type_tmp

        # 添加引用
        parameter_type_tmp += "&" * i["reference"]
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


# 将注释内容解析为说明字典
# 解析前: @brief Construct a Tensor from a buffer pointed to by `data` @note `from_blob` doesn’t copy or move data, Modifying the constructed tensor is equivalent to modifying the original data. @param data The pointer to the memory buffer. @param shape The dims of the tensor. @param dtype The data type of the tensor, should correspond to data type of`data`. See PD_FOR_EACH_DATA_TYPE in `phi/common/data_type.h` @param layout The data layout of the tensor. @param place The place where the tensor is located.If `place` is default value, it will be inferred from `data`,However, the feature is only supported on CPU or GPU.If `place` is not default value, make sure that `place` is equalto the place of `data` @param deleter A function or function object that will be called to free thememory buffer. @return A Tensor object constructed from the buffer
# 以@作为分隔符，索引关键字包括'brief'、'note'、'return'、'param'
# 解析后分别将对应关键字后的内容放入字典对应关键字后
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
