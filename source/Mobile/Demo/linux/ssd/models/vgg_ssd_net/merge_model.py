import paddle.v2 as paddle
from paddle.utils.merge_model import merge_v2_model
import vgg_ssd_net

if __name__ == "__main__":
    net = vgg_ssd_net.net_conf(mode='infer')
    param_file = "vgg_model-latest.tar.gz"
    output_file = "../vgg_ssd_net.paddle"
    merge_v2_model(net, param_file, output_file)
