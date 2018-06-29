import numpy as np
import gzip
import os
import argparse

import paddle.v2 as paddle
from paddle.v2.topology import Topology
from mobilenet_with_bn import mobile_net


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='merge batch normalization to Conv or Fc')
    parser.add_argument(
        '--source_model',
        help='the source model that should be merged',
        type=str)
    parser.add_argument(
        '--dest_model',
        help='the dest model name, we save our merged model of this name',
        type=str)
    return parser.parse_args()


class Merge_BN:
    def __init__(self, source_net, source_model, dest_model):
        self.source_proto = Topology(source_net).proto()
        self.source_layers = self.source_proto.layers

        with gzip.open(dest_model) as f:
            self.dest_param = paddle.parameters.Parameters.from_tar(f)

        with gzip.open(source_model) as f:
            self.source_param = paddle.parameters.Parameters.from_tar(f)

        self.dest_model = dest_model

    def fuse_param(self, current_layer, bn_layer):
        '''
        fuse the bn_layer' parameters to current_layer
        '''
        param_name = current_layer.inputs[0].input_parameter_name
        bias_name = current_layer.bias_parameter_name
        assert param_name, 'This layer(fc or exconv) should have parameters'
        bn_inputs = bn_layer.inputs

        a_bn_name = bn_inputs[0].input_parameter_name
        mean_bn_name = bn_inputs[1].input_parameter_name
        var_bn_name = bn_inputs[2].input_parameter_name
        b_bn_name = bn_layer.bias_parameter_name

        a_bn = self.source_param.get(a_bn_name)
        mean_bn = self.source_param.get(mean_bn_name)
        var_bn = self.source_param.get(var_bn_name)
        b_bn = self.source_param.get(b_bn_name)

        param = self.source_param.get(param_name)

        bias = np.zeros(a_bn.shape[1])
        if not bias_name:
            bias_name = param_name.split('.')[0] + '.wbias'
        else:
            bias = self.source_param.get(bias_name).reshape(1, -1)

        std_bn = np.float32(np.sqrt(np.add(var_bn, 1e-5)))
        tmp1 = np.float32(np.divide(a_bn, std_bn))
        bias = np.float32(
            np.add(np.multiply(np.subtract(bias, mean_bn), tmp1), b_bn))
        tmp1 = tmp1.reshape(tmp1.shape[1], -1)
        param = param.reshape((tmp1.shape[0], -1))
        param = np.float32(np.multiply(param, tmp1))

        if param_name in self.dest_param.names():
            print('Merge ' + param_name + ' SUCCEED ')
            param_shape = self.dest_param.get(param_name).shape
            self.dest_param.set(param_name, param.reshape(param_shape))
        if bias_name in self.dest_param.names():
            print('Merge ' + bias_name + ' SUCCEED ')
            bias_shape = self.dest_param.get(bias_name).shape
            self.dest_param.set(bias_name, bias.reshape(bias_shape))

    def save_layer_without_bn(self, current_layer):
        '''
        deal with the layer which has parameter but without batch_norm
        '''
        param_name = current_layer.inputs[0].input_parameter_name
        bias_name = current_layer.bias_parameter_name
        assert param_name, 'This layer(fc or exconv) should have parameters'

        param = self.source_param.get(param_name)
        if param_name in self.dest_param.names():
            print('Merge ' + param_name + ' SUCCEED ')
            self.dest_param.set(param_name, param)

        if bias_name:
            bias = self.source_param.get(bias_name)
            if bias_name in self.dest_param.names():
                print('Merge ' + bias_name + ' SUCCEED ')
                self.dest_param.set(bias_name, bias)

    def merge(self):
        '''
        Merge batch norm
        Currently, the default layer with parameters are fc and exconv layers.
        '''
        layer_num = len(self.source_layers)
        i = 0

        while i < layer_num:
            current_layer = self.source_layers[i]

            if current_layer.type in ['exconv', 'fc']:
                if (i + 1 < layer_num and
                        self.source_layers[i + 1].type == 'batch_norm'):
                    self.fuse_param(current_layer, self.source_layers[i + 1])
                    i = i + 2
                    continue
                else:
                    self.save_layer_without_bn(current_layer)
            i = i + 1
        with gzip.open(self.dest_model, 'w') as f:
            self.dest_param.to_tar(f)


if __name__ == "__main__":
    args = parse_args()

    img_size = 3 * 224 * 224
    class_num = 102

    # the net with batch norm
    net = mobile_net(img_size, class_num)

    base_path = os.path.dirname(os.path.realpath(__file__))
    # Attr: net, source_model_path, target_model_path
    mb = Merge_BN(net,
                  os.path.join(base_path, 'models', args.source_model),
                  os.path.join(base_path, 'models', args.dest_model))
    mb.merge()
    print('Merged SUCCESS!\n')

    print('Then Run verify.py to verify the correctness of merged model :)')
