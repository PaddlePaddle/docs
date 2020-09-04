import numpy as np
import argparse
import cv2

from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


# this is a simple resnet block for dynamci test.
def create_predictor(args):
    config = AnalysisConfig('./model')
    config.switch_use_feed_fetch_ops(False)
    config.enable_memory_optim()
    config.enable_use_gpu(100, 0)

    # using dynamic shpae mode, the max_batch_size will be ignored.
    config.enable_tensorrt_engine(
        workspace_size=1 << 30,
        max_batch_size=1,
        min_subgraph_size=5,
        precision_mode=AnalysisConfig.Precision.Float32,
        use_static=False,
        use_calib_mode=False)

    head_number = 12

    names = [
        "placeholder_0", "placeholder_1", "placeholder_2", "stack_0.tmp_0"
    ]
    min_input_shape = [1, 1, 1]
    max_input_shape = [100, 128, 1]
    opt_input_shape = [10, 60, 1]

    config.set_trt_dynamic_shape_info({
        names[0]: min_input_shape,
        names[1]: min_input_shape,
        names[2]: min_input_shape,
        names[3]: [1, head_number, 1, 1]
    }, {
        names[0]: max_input_shape,
        names[1]: max_input_shape,
        names[2]: max_input_shape,
        names[3]: [100, head_number, 128, 128]
    }, {
        names[0]: opt_input_shape,
        names[1]: opt_input_shape,
        names[2]: opt_input_shape,
        names[3]: [10, head_number, 60, 60]
    })
    predictor = create_paddle_predictor(config)
    return predictor


def run(predictor, data):
    # copy data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_tensor(name)
        input_tensor.reshape(data[i].shape)
        input_tensor.copy_from_cpu(data[i].copy())

    # do the inference
    predictor.zero_copy_run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_tensor(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Model filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="",
        help="Parameter filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Model dir, If you load a non-combined model, specify the directory of the model."
    )
    parser.add_argument(
        "--use_gpu", type=int, default=0, help="Whether use gpu.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pred = create_predictor(args)
    in1 = np.ones((1, 128, 1)).astype(np.int64)
    in2 = np.ones((1, 128, 1)).astype(np.int64)
    in3 = np.ones((1, 128, 1)).astype(np.int64)
    in4 = np.ones((1, 128, 1)).astype(np.float32)
    result = run(pred, [in1, in2, in3, in4])
    print(result)
