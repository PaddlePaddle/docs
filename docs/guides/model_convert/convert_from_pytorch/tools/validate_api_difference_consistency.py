import os
from pathlib import Path

from utils import extract_no_need_convert_list, load_mapping_json

# Note(littleherozzzx): get_api_difference_info.py not support parse overloaded
# functions currently. Currently, we hard code the check of overloaded functions
# in this file.

TENSOR_SPLIT_SIG_ARGS = [("sections",), ("indices",)]
SPLIT_SIG_ARGS = [("input", "sections"), ("input", "indices")]
REDUCE_ARGS = [("dim", "keepdim"), ("other",)]
OVERLOADED_APIS = {
    "torch.Tensor.dsplit": {"src_args": TENSOR_SPLIT_SIG_ARGS},
    "torch.Tensor.hsplit": {"src_args": TENSOR_SPLIT_SIG_ARGS},
    "torch.dsplit": {"src_args": SPLIT_SIG_ARGS},
    "torch.hsplit": {"src_args": SPLIT_SIG_ARGS},
    "torch.vsplit": {"src_args": SPLIT_SIG_ARGS},
    "torch.Tensor.vsplit": {"src_args": TENSOR_SPLIT_SIG_ARGS},
    "torch.Tensor.max": {"src_args": REDUCE_ARGS},
    "torch.Tensor.min": {"src_args": REDUCE_ARGS},
}


ALLOW_MISSING_DIFF_DOCS = [
    # Flags 类API，使用 Mock 实现，对 Paddle 行为无影响，归类为可删去。
    "torch.backends.cuda.matmul.allow_tf32",
    "torch.backends.cudnn.allow_tf32",
    "torch.backends.cudnn.benchmark",
    "torch.backends.cudnn.deterministic",
    "torch.backends.cudnn.enabled",
    # 不支持转换的 API
    "torch.Tensor.rename",
]


def validate_api_mappings():
    sum = 0
    # 获取当前脚本所在目录
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # 加载API差异数据
    api_diff = load_mapping_json(current_dir / "api_difference_info.json")
    api_diff_map: dict[str:dict] = {}

    # 加载API映射数据
    api_map = load_mapping_json(current_dir / "api_mapping.json")

    attr_map = load_mapping_json(current_dir / "attribute_mapping.json")

    api_map = api_map | attr_map

    for api in ALLOW_MISSING_DIFF_DOCS:
        api_map.pop(api, None)

    no_need_list = extract_no_need_convert_list(
        str(current_dir) + "/global_var.py"
    )

    # 准备错误报告文件
    error_file = current_dir / "validate_api_difference_consistency_error.txt"

    with open(error_file, "w", encoding="utf-8") as err_file:
        # 任务0: 检查 api_diff 文档中 src_api 是唯一的，并构造映射字典
        for entry in api_diff:
            src_api = entry["src_api"]
            if src_api in api_diff_map:
                err_file.write(
                    f"ERROR: api_difference_info entry '{src_api}' is not unique\n"
                )
                sum += 1
            else:
                api_diff_map[src_api] = entry

        # 任务1: 检查api_mapping中存在Matcher的条目是否在差异文档中有对应
        for api_key, api_value in api_map.items():
            if (
                api_value.get("Matcher") is not None
                and api_key not in no_need_list
            ):
                if api_key not in api_diff_map:
                    err_file.write(
                        f"WARNING: api_mapping entry '{api_key}' not found in api_difference_info.json\n"
                    )
                    sum += 1

        # 任务2: 检查差异文档中的dst_api是否与api_mapping中paddle_api一致
        for diff_entry in api_diff:
            # 如果文档里没写目标 API（可能是组合实现），跳过
            if "dst_api" not in diff_entry:
                continue

            torch_api = diff_entry["src_api"]
            doc_target = diff_entry["dst_api"]  # 文档里的目标

            # 1. 检查代码规则库里有没有这个 torch_api
            # 注意：这里要查 api_map (规则)，而不是 api_diff_map (文档自己)
            if torch_api not in api_map:
                # 这种情况（文档有，规则无）通常不在这里处理，或者仅作为 Info
                continue

            rule_entry = api_map[torch_api]

            # 2. 获取代码规则里的目标 API
            # 注意：api_mapping.json 里的键名是 'paddle_api'
            code_target = rule_entry.get("paddle_api")

            # 3. 如果代码里有明确的目标 API，检查是否一致
            if code_target and code_target != doc_target:
                err_file.write(
                    f"ERROR: Mapping Conflict for '{torch_api}':\n"
                    f"    Document says -> {doc_target}\n"
                    f"    Code rule says -> {code_target}\n"
                )
                sum += 1

        # for entry in api_diff:
        #     if "dst_api" not in entry:
        #         continue

        #     found = False
        #     for api_key, api_value in api_map.items():
        #         if api_value.get("paddle_api") == entry["dst_api"]:
        #             found = True
        #             break

        #     if not found:
        #         err_file.write(
        #             f"WARNING: api_difference_info entry '{entry['src_api']}' has dst_api '{entry['dst_api']}' not found in api_mapping.json\n"
        #         )
        #         sum += 1

        # 任务3: 检查api_mapping中的kwargs_change是否在api_difference_info中
        for api_key, api_value in api_map.items():
            if "kwargs_change" not in api_value or api_key in no_need_list:
                continue
            if api_key in OVERLOADED_APIS:  # 重载暂不支持解析 diff 文档
                continue
            entry = api_diff_map.get(api_key)
            if entry is None:
                continue
            if entry["mapping_type"] == "组合替代实现":
                continue  # 组合替代实现不需要参数映射

            for src_arg, dst_arg in api_value["kwargs_change"].items():
                found = False
                if "args_mapping" not in entry:
                    continue
                for mapping in entry["args_mapping"]:
                    if mapping["dst_arg"] == "-":
                        mapping["dst_arg"] = ""
                    if "," in mapping["dst_arg"]:
                        mapping["dst_arg"] = (
                            mapping["dst_arg"]
                            .replace(" ", "")
                            .replace("\t", "")
                            .split(",")
                        )
                    if (
                        mapping["src_arg"] == src_arg
                        and mapping["dst_arg"] == dst_arg
                    ):
                        found = True
                        break
                if found:
                    break

                if not found:
                    err_file.write(
                        f"ERROR: Parameter mapping '{src_arg} -> {dst_arg}' in api_mapping for '{api_key}' not found in api_difference_info.json\n"
                    )
                    sum += 1

        # -----------------------------------------------------------------
        # 任务4: 检查api_mapping中的args_list是否都在api_difference_info的src_signature中
        #       (支持 OVERLOADED_APIS)
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # 任务4: 检查api_mapping中的args_list是否都在api_difference_info的src_signature中
        #       (支持 OVERLOADED_APIS，且双方统一清洗参数名)
        # -----------------------------------------------------------------
        for api_key, api_value in api_map.items():
            # 跳过无需检查的情况
            if "args_list" not in api_value or api_key in no_need_list:
                continue

            args_list_in_mapping = api_value["args_list"]

            # 定义清洗函数：去掉参数名前的 * 或 **，保留纯参数名
            def normalize_param_name(name):
                if name == "*":
                    return "*"
                return name.lstrip("*")

            # -------------------------------------------
            # 分支 A：硬编码的重载 API 检查
            # -------------------------------------------
            if api_key in OVERLOADED_APIS:
                valid_signatures_groups = OVERLOADED_APIS[api_key].get(
                    "src_args", []
                )

                # 1. 生成全集 (Reference): 清洗定义中的参数
                allowed_args_superset = set()
                for group in valid_signatures_groups:
                    current_group = (
                        (group,) if isinstance(group, str) else group
                    )
                    cleaned_group = {
                        normalize_param_name(arg) for arg in current_group
                    }
                    allowed_args_superset.update(cleaned_group)

                # 2. 准备检查对象 (Target): 清洗 Mapping 中的参数 【修复点】
                mapping_args_set = {
                    normalize_param_name(arg) for arg in args_list_in_mapping
                }

                # 3. 集合求差
                unknown_args = mapping_args_set - allowed_args_superset

                if unknown_args:
                    err_file.write(
                        f"ERROR: Args list {args_list_in_mapping} in api_mapping for OVERLOADED API '{api_key}' "
                        f"contains arguments (normalized: {unknown_args}) which are NOT found in any signature defined in OVERLOADED_APIS.\n"
                    )
                    sum += 1
                continue

            # -------------------------------------------
            # 分支 B：基于 JSON 文档的普通 API 检查
            # -------------------------------------------
            entry = api_diff_map.get(api_key)
            if entry is None:
                err_file.write(
                    f"ERROR: api_mapping for '{api_key}' not found in api_difference_info.json\n"
                )
                sum += 1
                continue

            if entry.get("mapping_type") == "组合替代实现":
                continue

            if "src_signature" not in entry or not entry["src_signature"]:
                err_file.write(
                    f"ERROR: api_difference_info for '{api_key}' has no src_signature\n"
                )
                sum += 1
                continue

            # 1. 提取并清洗文档中的签名 (Reference)
            src_signature = entry["src_signature"][0]
            if "args" not in src_signature:
                err_file.write(
                    f"ERROR: api_difference_info for '{api_key}' has no args in src_signature\n"
                )
                sum += 1
                continue

            src_args = [
                normalize_param_name(arg["arg_name"])
                for arg in src_signature["args"]
            ]

            # 2. 逐个检查 Mapping 中的参数 (Target)
            for arg in args_list_in_mapping:
                # 【修复点】：先清洗，再比较
                normalized_arg = normalize_param_name(arg)

                if normalized_arg not in src_args:
                    err_file.write(
                        f"ERROR: Parameter '{arg}' (normalized: '{normalized_arg}') in api_mapping for '{api_key}' "
                        f"not found in api_difference_info's src_signature: {src_args}\n"
                    )
                    sum += 1
    print(
        f"{sum} api error found in api_mapping.json and api_difference_info.json"
    )
    return sum


if __name__ == "__main__":
    validate_api_mappings()
