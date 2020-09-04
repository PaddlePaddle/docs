#coding: utf-8
from __future__ import print_function
import numpy as np
import paddle
import argparse
import reader
import sys

from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


def parse_args():
    """
    Parsing the input parameters.
    """
    parser = argparse.ArgumentParser("Inference for lexical analyzer.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="elmo",
        help="The folder where the test data is located.")
    parser.add_argument(
        "--testdata_dir",
        type=str,
        default="elmo_data/dev",
        help="The folder where the test data is located.")
    parser.add_argument(
        "--use_gpu",
        type=int,
        default=False,
        help="Whether or not to use GPU. 0-->CPU 1-->GPU")
    parser.add_argument(
        "--word_dict_path",
        type=str,
        default="elmo_data/vocabulary_min5k.txt",
        help="The path of the word dictionary.")
    parser.add_argument(
        "--label_dict_path",
        type=str,
        default="elmo_data/tag.dic",
        help="The path of the label dictionary.")
    parser.add_argument(
        "--word_rep_dict_path",
        type=str,
        default="elmo_data/q2b.dic",
        help="The path of the word replacement Dictionary.")

    args = parser.parse_args()
    return args


def to_lodtensor(data):
    """
    Convert data in list into lodtensor.
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    return flattened_data, [lod]


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

    predictor = create_paddle_predictor(config)
    return predictor


def run(predictor, datas, lods):
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_tensor(name)
        input_tensor.reshape(datas[i].shape)
        input_tensor.copy_from_cpu(datas[i].copy())
        input_tensor.set_lod(lods[i])

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


if __name__ == '__main__':

    args = parse_args()
    word2id_dict = reader.load_reverse_dict(args.word_dict_path)
    label2id_dict = reader.load_reverse_dict(args.label_dict_path)
    word_rep_dict = reader.load_dict(args.word_rep_dict_path)
    word_dict_len = max(map(int, word2id_dict.values())) + 1
    label_dict_len = max(map(int, label2id_dict.values())) + 1

    pred = create_predictor(args)

    test_data = paddle.batch(
        reader.file_reader(args.testdata_dir, word2id_dict, label2id_dict,
                           word_rep_dict),
        batch_size=1)
    batch_id = 0
    id2word = {v: k for k, v in word2id_dict.items()}
    id2label = {v: k for k, v in label2id_dict.items()}
    for data in test_data():
        batch_id += 1
        word_data, word_lod = to_lodtensor(list(map(lambda x: x[0], data)))
        target_data, target_lod = to_lodtensor(list(map(lambda x: x[1], data)))
        result_list = run(pred, [word_data, target_data],
                          [word_lod, target_lod])
        number_infer = np.array(result_list[0])
        number_label = np.array(result_list[1])
        number_correct = np.array(result_list[2])
        lac_result = ""
        for i in range(len(data[0][0])):
            lac_result += id2word[data[0][0][i]] + '/' + id2label[np.array(
                result_list[3]).tolist()[i][0]] + " "
        print("%d sample's result:" % batch_id, lac_result)
        if batch_id >= 10:
            exit()
