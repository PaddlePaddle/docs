# Python 预测 API介绍

Fluid提供了高度优化的[C++预测库](./native_infer.html)，为了方便使用，我们也提供了C++预测库对应的Python接口，两者含义完全相同，下面是详细的使用说明


## PaddleTensor

`PaddleTensor`是预测库输入和输出的数据结构，包括以下字段

* `name`(str): 指定输入的名称
* `shape`(tuple|list): Tensor的shape
* `data`(PaddleBuf): Tensor的数据，存储在`PaddleBuf`中，
* `dtype`(PaddleDType): Tensor的类型

## PaddleBuf

`PaddleBuf`定义了`PaddleTensor`的存储结构，创建`PaddleBuf`:

``` python
int64_buf = PaddleBuf([1, 2, 3, 4])
float_buf = PaddleBuf([1., 2., 3., 4.])
```

`PadleBuf`包括以下方法

* `resize`: 重新分配内存，单位为byte
* `reset`: 重新设置数据
* `empty`: buffer是否为空
* `float_data`: 将数据转为float型的list返回
* `int64_data`: 将数据转为int64型的list返回
* `length`: 内存大小，单位为byte

## PaddleDType

`PaddleDType`定义了`PaddleTensor`的类型，包括

* `PaddleDType.INT64`: 64位整型
* `PaddleDType.FLOAT32`: 32位浮点型

## AnalysisConfig

`AnalysisConfig`是创建预测引擎的配置，主要包括以下方法  

* `set_model`: 设置模型的路径
* `model_dir`: 返回模型路径
* `enable_use_gpu`: 设置GPU显存(单位M)和ID
* `disable_gpu`: 禁用GPU
* `gpu_device_id`: 返回使用的GPU ID
* `switch_ir_optim`: IR优化(默认开启)
* `enable_tensorrt_engine`: 启用TensorRT
* `enable_mkldnn`: 启用MKLDNN


## PaddlePredictor

`PaddlePredictor`是运行预测的引擎，下面是创建和使用的说明   

``` python
# 创建预测引擎
config = AnalysisConfig(model_dir)
config.enable_use_gpu(200, 0) # 200M显存, 设备id为0
config.enable_tensorrt_engine() # 打开TensorRT

predictor = create_paddle_predictor(config)

# 设置输入
x = fluid.core.PaddleTensor()
# x.name = ...
# x.shape = ...
# x.data = ...
# x.dtype = ...

y = fluid.core.PaddleTensor()
# y.name = ...
# y.shape = ...
# y.data = ...
# y.dtype = ...


# 运行预测引擎得到结果，返回值是一个PaddleTensor的list
results = predictor.run([x, y])

# 获得 results，并应用到自己的应用中
```

**Python API 相关接口与 C++ API 完全对应，可以对照查阅**

## 完整使用示例

下面是一个完整的resnet50预测示例

下载[resnet50模型](http://paddle-inference-dist.bj.bcebos.com/resnet50_model.tar.gz)并解压，运行如下命令将会调用预测引擎

``` bash
python resnet50_infer.py --model_dir model --prog_file model --params_file params --batch_size 2
```

`resnet50_infer.py` 的内容是

``` python
import argparse
import numpy as np

from paddle.fluid.core import PaddleBuf
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


def main():
    args = parse_args()

    # Set config
    config = AnalysisConfig(args.model_dir)
    config.disable_gpu()

    # Create PaddlePredictor
    predictor = create_paddle_predictor(config)

    # Set inputs
    inputs = fake_input(args.batch_size)

    # Infer
    outputs = predictor.run(inputs)

    # parse outputs
    output = outputs[0]
    print(output.name)
    output_data = output.data.float_data()
    assert len(output_data) == 512 * args.batch_size
    for i in range(args.batch_size):
        print(np.argmax(output_data[i * 512:(i + 1) * 512]))


def fake_input(batch_size):
    image = PaddleTensor()
    image.name = "data"
    image.shape = [batch_size, 3, 318, 318]
    image.dtype = PaddleDType.FLOAT32
    image.data = PaddleBuf(
        np.random.randn(*image.shape).flatten().astype("float32").tolist())
    return [image]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="model dir")
    parser.add_argument("--prog_file", type=str, help="program filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")

    return parser.parse_args()


if __name__ == "__main__":
    main()    
```
