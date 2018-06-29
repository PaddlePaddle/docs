import paddle.v2 as paddle
from paddle.utils.merge_model import merge_v2_model

# import network configuration
from mobilenet import mobile_net

if __name__ == "__main__":
    image_size = 224
    num_classes = 102
    net = mobile_net(3 * image_size * image_size, num_classes, 1.0)
    param_file = './mobilenet_flowers102.tar.gz'
    output_file = './mobilenet_flowers102.paddle'
    merge_v2_model(net, param_file, output_file)
