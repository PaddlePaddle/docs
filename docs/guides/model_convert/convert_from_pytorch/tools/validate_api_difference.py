import json
import os
from pathlib import Path

from utils import extract_no_need_convert_list


def validate_api_mappings():
    sum = 0
    # 获取当前脚本所在目录
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # 加载API差异数据
    with open(
        current_dir / "api_difference_info.json", "r", encoding="utf-8"
    ) as f:
        api_diff = json.load(f)

    # 加载API映射数据
    with open(current_dir / "api_mapping.json", "r", encoding="utf-8") as f:
        api_map = json.load(f)

    no_need_list = extract_no_need_convert_list(
        str(current_dir) + "/global_var.py"
    )

    # 准备错误报告文件
    error_file = current_dir / "validate_api_difference_error.txt"

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

        # 任务3: 检查参数映射表与api_mapping中的kwargs_change是否一致
        for entry in api_diff:
            if "args_mapping" not in entry:
                continue
            api_error = False

            for mapping in entry["args_mapping"]:
                src_arg = mapping["src_arg"]
                dst_arg = mapping["dst_arg"]
                if src_arg == dst_arg:
                    continue
                # 如果 dst_arg 不是 "-"，则检查 kwargs_change
                if src_arg == "-":
                    # 这种情况表示在 PyTorch 中没有这个参数，但在 Paddle 中有，我们跳过
                    continue
                if dst_arg == "-":
                    dst_arg = ""

                found = False
                for api_value in api_map.values():
                    if "kwargs_change" in api_value:
                        if (
                            src_arg in api_value["kwargs_change"]
                            and api_value["kwargs_change"][src_arg] == dst_arg
                        ):
                            found = True
                            break

                if not found:
                    if "unsupport_args" not in api_value:
                        api_error = True
                        err_file.write(
                            f"ERROR: src_arg '{src_arg}' with dst_arg '-' in api_difference_info for '{entry['src_api']}' not found in api_mapping.json\n"
                        )
                    elif src_arg not in api_value["unsupport_args"]:
                        api_error = True
                        err_file.write(
                            f"ERROR: src_arg '{src_arg}' with dst_arg '-' in api_difference_info for '{entry['src_api']}' not found in api_mapping.json\n"
                        )

            if api_error:
                sum += 1

        # 任务4: 检查api_mapping中的args_list与api_difference_info中的函数签名参数列表是否一致
        for entry in api_diff:
            # 跳过没有src_signature的条目
            if "src_signature" not in entry or not entry["src_signature"]:
                continue

            # 提取第一个src_signature的参数名列表
            src_signature = entry["src_signature"][0]
            if "args" not in src_signature:
                continue
            src_args = [arg["arg_name"] for arg in src_signature["args"]]

            # 检查api_map中是否有这个src_api
            if entry["src_api"] not in api_map:
                err_file.write(
                    f"ERROR: api_mapping for '{entry['src_api']}' not found in api_mapping.json, so cannot check args_list\n"
                )
                sum += 1
                continue

            api_value = api_map[entry["src_api"]]
            # 检查args_list是否存在
            if "args_list" not in api_value:
                err_file.write(
                    f"ERROR: api_mapping for '{entry['src_api']}' has no args_list\n"
                )
                sum += 1
                continue

            api_args_list = api_value["args_list"]

            # 比较参数列表
            if src_args != api_args_list:
                err_file.write(
                    f"ERROR: Parameter list mismatch for '{entry['src_api']}': "
                    f"api_mapping has {api_args_list}, but api_difference_info has {src_args}\n"
                )
                sum += 1

    print(
        f"{sum} api error found in api_mapping.json and api_difference_info.json"
    )
    return sum


if __name__ == "__main__":
    validate_api_mappings()
