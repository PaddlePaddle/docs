import os
from pathlib import Path

from utils import extract_no_need_convert_list, load_mapping_json

# Note(littleherozzzx): get_api_difference_info.py not support parse overloaded
# functions currently. Currently, we hard code the check of overloaded functions
# in this file.

OVERLOADED_APIS = {
    "torch.Tensor.dsplit": {
        "src_args": [
            ("sections"),
            ("indices"),
        ]
    },
    "torch.Tensor.hsplit": {
        "src_args": [
            ("sections"),
            ("indices"),
        ]
    },
    "torch.dsplit": {
        "src_args": [
            ("input", "sections"),
            ("input", "indices"),
        ]
    },
    "torch.hsplit": {
        "src_args": [
            ("input", "sections"),
            ("input", "indices"),
        ]
    },
    "torch.vsplit": {
        "src_args": [
            ("input", "sections"),
            ("input", "indices"),
        ]
    },
    "torch.Tensor.max": {
        "src_args": [
            ("dim", "keepdim"),
            ("other"),
        ]
    },
    "torch.Tensor.min": {
        "src_args": [
            ("dim", "keepdim"),
            ("other"),
        ]
    },
    "torch.linalg.matrix_rank": {
        "src_args": [
            ("A", "*", "atol", "rtol", "hermitian", "out"),
            ("input", "*", "atol", "rtol", "hermitian", "out"),
            ("input", "tol", "hermitian", "*", "out"),
        ]
    },
    "torch.std_mean": {
        "src_args": [
            ("input", "dim", "unbiased", "keepdim"),
            ("input", "dim", "*", "correction", "keepdim"),
            ("input", "unbiased"),
            ("input", "dim", "unbiased", "keepdim"),
        ]
    },
    "torch.var_mean": {
        "src_args": [
            ("input", "dim", "unbiased", "keepdim"),
            ("input", "dim", "*", "correction", "keepdim"),
            ("input", "unbiased"),
            ("input", "dim", "unbiased", "keepdim"),
        ]
    },
    "torchvision.models.alexnet": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.densenet121": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.densenet161": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.densenet169": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.densenet201": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.googlenet": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.inception_v3": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.mobilenet_v2": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.mobilenet_v3_large": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.mobilenet_v3_small": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.resnet101": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.resnet152": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.resnet18": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.resnet34": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.resnet50": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.resnext101_64x4d": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.resnext50_32x4d": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.shufflenet_v2_x0_5": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.shufflenet_v2_x1_0": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.shufflenet_v2_x1_5": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.shufflenet_v2_x2_0": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.squeezenet1_0": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.squeezenet1_1": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.vgg11": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.vgg11_bn": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.vgg13": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.vgg13_bn": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.vgg16": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.vgg16_bn": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.vgg19": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.vgg19_bn": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.wide_resnet101_2": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
    "torchvision.models.wide_resnet50_2": {
        "src_args": [
            ("*", "weights", "progress", "**kwargs"),
            ("*", "pretrained", "progress", "**kwargs"),
        ]
    },
}


ALLOW_MISSING_DIFF_DOCS = [
    # Flags 类API，使用 Mock 实现，对 Paddle 行为无影响，归类为可删去。
    "torch.backends.cuda.matmul.allow_tf32",
    "torch.backends.cudnn.allow_tf32",
    "torch.backends.cudnn.benchmark",
    "torch.backends.cudnn.deterministic",
    "torch.backends.cudnn.enabled",
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

        # 任务4: 检查api_mapping中的args_list是否都在api_difference_info的src_signature中
        for api_key, api_value in api_map.items():
            if "args_list" not in api_value or api_key in no_need_list:
                continue

            # 获取api_difference_info中对应的entry
            entry = api_diff_map.get(api_key)
            if entry is None:
                err_file.write(
                    f"ERROR: api_mapping for '{api_key}' not found in api_difference_info.json, so cannot check args_list\n"
                )
                sum += 1
                continue
            if entry["mapping_type"] == "组合替代实现":
                continue  # 组合替代实现不记录参数信息

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
