import os
from pathlib import Path

from utils import extract_no_need_convert_list, load_mapping_json


def validate_api_mappings():
    sum = 0
    # 获取当前脚本所在目录
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # 加载API差异数据
    api_diff = load_mapping_json(current_dir / "api_difference_info.json")

    # 加载API映射数据
    api_map = load_mapping_json(current_dir / "api_mapping.json")

    attr_map = load_mapping_json(current_dir / "attribute_mapping.json")

    api_map = api_map | attr_map

    no_need_list = extract_no_need_convert_list(
        str(current_dir) + "/global_var.py"
    )

    # 准备错误报告文件
    error_file = current_dir / "validate_api_difference_consistency_error.txt"

    # 任务1: 检查api_mapping中存在Matcher的条目是否在差异文档中有对应
    with open(error_file, "w", encoding="utf-8") as err_file:
        for api_key, api_value in api_map.items():
            if (
                api_value.get("Matcher") is not None
                and api_key not in no_need_list
            ):
                if not any(entry["src_api"] == api_key for entry in api_diff):
                    err_file.write(
                        f"WARNING: api_mapping entry '{api_key}' not found in api_difference_info.json\n"
                    )
                    sum += 1

        # 任务2: 检查差异文档中的dst_api是否与api_mapping中paddle_api一致
        for entry in api_diff:
            if "dst_api" not in entry:
                continue

            found = False
            for api_key, api_value in api_map.items():
                if api_value.get("paddle_api") == entry["dst_api"]:
                    found = True
                    break

            if not found:
                err_file.write(
                    f"WARNING: api_difference_info entry '{entry['src_api']}' has dst_api '{entry['dst_api']}' not found in api_mapping.json\n"
                )
                sum += 1

        # 任务3: 检查api_mapping中的kwargs_change是否在api_difference_info中
        for api_key, api_value in api_map.items():
            if "kwargs_change" not in api_value or api_key in no_need_list:
                continue

            for src_arg, dst_arg in api_value["kwargs_change"].items():
                found = False
                for entry in api_diff:
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

        # 任务4: 检查api_mapping中的args_list是否都在api_difference_info的src_signature中
        for api_key, api_value in api_map.items():
            if "args_list" not in api_value or api_key in no_need_list:
                continue

            # 获取api_difference_info中对应的entry
            entry = next((e for e in api_diff if e["src_api"] == api_key), None)
            if entry is None:
                err_file.write(
                    f"ERROR: api_mapping for '{api_key}' not found in api_difference_info.json, so cannot check args_list\n"
                )
                sum += 1
                continue

            if "src_signature" not in entry or not entry["src_signature"]:
                err_file.write(
                    f"ERROR: api_difference_info for '{api_key}' has no src_signature\n"
                )
                sum += 1
                continue

            # 提取第一个src_signature的参数名列表
            src_signature = entry["src_signature"][0]
            if "args" not in src_signature:
                err_file.write(
                    f"ERROR: api_difference_info for '{api_key}' has no args in src_signature\n"
                )
                sum += 1
                continue
            src_args = [arg["arg_name"] for arg in src_signature["args"]]

            # 检查api_mapping中的args_list是否都在src_args中
            for arg in api_value["args_list"]:
                if arg not in src_args:
                    err_file.write(
                        f"ERROR: Parameter '{arg}' in api_mapping for '{api_key}' not found in api_difference_info's src_signature\n"
                    )
                    sum += 1

    print(
        f"{sum} api error found in api_mapping.json and api_difference_info.json"
    )
    return sum


if __name__ == "__main__":
    validate_api_mappings()
