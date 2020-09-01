import numpy as np
import argparse
import cv2
from PIL import Image

from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

from utils import preprocess, draw_bbox


def create_predictor(args):
    if args.model_dir is not "":
        config = AnalysisConfig(args.model_dir)
    else:
        config = AnalysisConfig(args.model_file, args.params_file)

    config.switch_use_feed_fetch_ops(False)
    config.enable_memory_optim()
    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
    else:
        # If not specific mkldnn, you can set the blas thread.
        # The thread num should not be greater than the number of cores in the CPU.
        config.set_cpu_math_library_num_threads(4)
        #config.enable_mkldnn()

    predictor = create_paddle_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_tensor(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

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
    img_name = 'kite.jpg'
    save_img_name = 'res.jpg'
    im_size = 608
    pred = create_predictor(args)
    img = cv2.imread(img_name)
    data = preprocess(img, im_size)
    im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.int32)
    result = run(pred, [data, im_shape])
    img = Image.open(img_name).convert('RGB').resize((im_size, im_size))
    draw_bbox(img, result[0], save_name=save_img_name)
