# Python 预测 API介绍

Fluid提供了高度优化的[C++预测库](./native_infer.html)，为了方便使用，我们也提供了C++预测库对应的Python接口，下面是详细的使用说明。



## Python预测相关数据结构

使用Python预测API与C++预测API相似，主要包括`PaddleTensor`, `PaddleDType`, `AnalysisConfig`和`PaddlePredictor`，分别对应于C++ API中同名的类型。

### PaddleTensor

class paddle.fluid.core.PaddleTensor

`PaddleTensor`是预测库输入和输出的数据结构，包括以下字段

* `name`(str): 指定输入的名称
* `shape`(tuple|list): Tensor的shape
* `data`(numpy.ndarray): Tensor的数据，可在PaddleTensor构造的时候用`numpy.ndarray`直接传入
* `dtype`(PaddleDType): Tensor的类型
* `lod`(List[List[int]]): [LoD](../../../user_guides/howto/basic_concept/lod_tensor.html)信息

`PaddleTensor`包括以下方法

* `as_ndarray`: 返回`data`对应的numpy数组

#### 代码示例
``` python
tensor = PaddleTensor(name="tensor", data=numpy.array([1, 2, 3], dtype="int32"))
```
调用`PaddleTensor`的成员字段和方法输出如下：
``` python
>>> tensor.name
'tensor'
>>> tensor.shape
[3]
>>> tensor.dtype
PaddleDType.INT32
>>> tensor.lod
[]
>>> tensor.as_ndarray()
array([1, 2, 3], dtype=int32)
```


### PaddleDType

class paddle.fluid.core.PaddleTensor

`PaddleDType`定义了`PaddleTensor`的数据类型，由传入`PaddleTensor`的numpy数组类型确定，包括以下成员

* `INT64`: 64位整型
* `INT32`: 32位整型
* `FLOAT32`: 32位浮点型

### AnalysisConfig

class paddle.fluid.core.AnalysisConfig

`AnalysisConfig`是创建预测引擎的配置，提供了模型路径设置、预测引擎运行设备选择以及多种优化预测流程的选项，主要包括以下方法  

* `set_model`: 设置模型的路径
* `model_dir`: 返回模型文件夹路径
* `prog_file`: 返回模型文件路径
* `params_file`: 返回参数文件路径
* `enable_use_gpu`: 设置GPU显存(单位M)和Device ID
* `disable_gpu`: 禁用GPU
* `gpu_device_id`: 返回使用的GPU ID
* `switch_ir_optim`: IR优化(默认开启)
* `enable_tensorrt_engine`: 开启TensorRT
* `enable_mkldnn`: 开启MKLDNN
* `disable_glog_info`: 禁用预测中的glog日志
* `delete_pass`: 预测的时候删除指定的pass
#### 代码示例
设置模型和参数路径有两种形式：
* 当模型文件夹下存在一个模型文件和多个参数文件时，传入模型文件夹路径，模型文件名默认为`__model__`
``` python
config = AnalysisConfig("./model")
```
* 当模型文件夹下只有一个模型文件和一个参数文件时，传入模型文件和参数文件路径
``` python
config = AnalysisConfig("./model/model", "./model/params")
```
使用`set_model`方法设置模型和参数路径方式同上

其他预测引擎配置选项示例如下
``` python
config.enable_use_gpu(100, 0) # 初始化100M显存，使用gpu id为0
config.gpu_device_id()        # 返回正在使用的gpu id
config.disable_gpu()		  # 禁用gpu
config.switch_ir_optim(True)  # 开启IR优化
config.enable_tensorrt_engine(precision_mode=AnalysisConfig.Precision.Float32,
                              use_calib_mode=True) # 开启TensorRT预测，精度为fp32，开启int8离线量化
config.enable_mkldnn()		  # 开启MKLDNN
```



### PaddlePredictor

class paddle.fluid.core.PaddlePredictor

`PaddlePredictor`是运行预测的引擎，由`paddle.fluid.core.create_paddle_predictor(config)`创建，主要提供以下方法

* `run`: 输入和返回值均为`PaddleTensor`列表类型，功能为运行预测引擎，返回预测结果

#### 代码示例

``` python
# 设置完AnalysisConfig后创建预测引擎PaddlePredictor
predictor = create_paddle_predictor(config)

# 设置输入
x = numpy.array([1, 2, 3], dtype="int64")
x_t = fluid.core.PaddleTensor(x)

y = numpy.array([4], dtype = "int64")
y_t = fluid.core.PaddleTensor(y)

# 运行预测引擎得到结果，返回值是一个PaddleTensor的列表
results = predictor.run([x_t, y_t])

# 获得预测结果，并应用到自己的应用中
```

### 使用ZeroCopyTensor管理输入/输出

`ZeroCopyTensor`是`AnalysisPredictor`的一种输入/输出数据结构，与`PaddleTensor`等同。`ZeroCopyTensor`相比于`PaddleTensor`，可以避免预测时候准备输入以及获取输出时多余的数据拷贝，提高预测性能。

注意: 需要注意的是，使用`ZeroCopyTensor`，务必在创建`config`时设置`config.switch_use_feed_fetch_ops(False)`用于显式地在模型运行的时候删去`feed`和`fetch`ops，不会影响模型的效果，但是能提升性能。

``` python
# 创建predictor
predictor = create_paddle_predictor(config)

# 获取输入的名称
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_tensor(input_names[0])

# 设置输入
fake_input = numpy.random.randn(1, 3, 318, 318).astype("float32")
input_tensor.copy_from_cpu(fake_input)

# 运行predictor
predictor.zero_copy_run()

# 获取输出
output_names = predictor.get_output_names()
output_tensor = predictor.get_output_tensor(output_names[0])
output_data = output_tensor.copy_to_cpu() # numpy.ndarray类型
```

### AnalysisPredictor

class paddle.fluid.core.AnalysisPredictor

`AnalysisPredictor`是运行预测的引擎，继承于`PaddlePredictor`，同样是由`paddle.fluid.core.create_paddle_predictor(config)`创建，主要提供以下方法

* `zero_copy_run()`: 运行预测引擎，返回预测结果
* `get_input_names()`: 获取输入的名称
* `get_input_tensor(input_name: str)`: 根据输入的名称获取对应的`ZeroCopyTensor`
* `get_output_names()`: 获取输出的名称
* `get_output_tensor(output_name: str)`: 根据输出的名称获取对应的`ZeroCopyTensor`

#### 代码示例

``` python
# 设置完AnalysisConfig后创建预测引擎PaddlePredictor
predictor = create_paddle_predictor(config)

# 获取输入的名称
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_tensor(input_names[0])

# 设置输入
fake_input = numpy.random.randn(1, 3, 318, 318).astype("float32")
input_tensor.reshape([1, 3, 318, 318])
input_tensor.copy_from_cpu(fake_input)

# 运行predictor
predictor.zero_copy_run()

# 获取输出
output_names = predictor.get_output_names()
output_tensor = predictor.get_output_tensor(output_names[0])
```

## 支持方法列表
* PaddleTensor
	* `as_ndarray() -> numpy.ndarray`
* ZeroCopyTensor
    * `copy_from_cpu(input: numpy.ndarray) -> None`
    * `copy_to_cpu() -> numpy.ndarray`
    * `reshape(input: numpy.ndarray|List[int]) -> None`
    * `shape() -> List[int]`
    * `set_lod(input: numpy.ndarray|List[List[int]]) -> None`
    * `lod() -> List[List[int]]`
    * `type() -> PaddleDType`
* AnalysisConfig
	* `set_model(model_dir: str) -> None`
	* `set_model(prog_file: str, params_file: str) -> None`
	* `model_dir() -> str`
	* `prog_file() -> str`
	* `params_file() -> str`
	* `enable_use_gpu(memory_pool_init_size_mb: int, device_id: int) -> None`
	* `gpu_device_id() -> int`
	* `switch_ir_optim(x: bool = True) -> None`
	* `enable_tensorrt_engine(workspace_size: int = 1 << 20,
	                          max_batch_size: int,
                              min_subgraph_size: int,
                              precision_mode: AnalysisConfig.precision,
                              use_static: bool,
                              use_calib_mode: bool) -> None`
	* `enable_mkldnn() -> None`
    * `disable_glog_info() -> None`
    * `delete_pass(pass_name: str) -> None`
* PaddlePredictor
	* `run(input: List[PaddleTensor]) -> List[PaddleTensor]`
* AnalysisPredictor
    * `zero_copy_run() -> None`
    * `get_input_names() -> List[str]`
    * `get_input_tensor(input_name: str) -> ZeroCopyTensor`
    * `get_output_names() -> List[str]`
    * `get_output_tensor(output_name: str) -> ZeroCopyTensor`

可参考对应的[C++预测接口](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/inference_api.cc)，其中定义了每个接口的参数和返回值

## 完整使用示例

下面是使用Fluid Python API进行预测的一个完整示例，使用resnet50模型

下载[resnet50模型](http://paddle-inference-dist.bj.bcebos.com/resnet50_model.tar.gz)并解压，运行如下命令将会调用预测引擎

``` bash
python resnet50_infer.py --model_file ./model/model --params_file ./model/params --batch_size 2
```

`resnet50_infer.py` 的内容是

### PaddleTensor的完整使用示例

``` python
import argparse
import numpy as np

from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


def main():
    args = parse_args()

    # 设置AnalysisConfig
    config = AnalysisConfig(args.model_file, args.params_file)
    config.disable_gpu()

    # 创建PaddlePredictor
    predictor = create_paddle_predictor(config)

    # 设置输入，此处以随机输入为例，用户可自行输入真实数据
    inputs = fake_input(args.batch_size)

    # 运行预测引擎
    outputs = predictor.run(inputs)
    output_num = 512

    # 获得输出并解析
    output = outputs[0]
    print(output.name)
    output_data = output.as_ndarray() #return numpy.ndarray
    assert list(output_data.shape) == [args.batch_size, output_num]
    for i in range(args.batch_size):
        print(np.argmax(output_data[i]))


def fake_input(batch_size):  
    shape = [batch_size, 3, 318, 318]
    data = np.random.randn(*shape).astype("float32")
    image = PaddleTensor(data)
    return [image]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")

    return parser.parse_args()


if __name__ == "__main__":
    main()  
```

### ZeroCopyTensor的完整使用示例

``` python
import argparse
import numpy as np
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


def main():
    args = parse_args()

    # 设置AnalysisConfig
    config = set_config(args)

    # 创建PaddlePredictor
    predictor = create_paddle_predictor(config)

    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])

    # 设置输入
    fake_input = np.random.randn(1, 3, 318, 318).astype("float32")
    input_tensor.reshape([1, 3, 318, 318])
    input_tensor.copy_from_cpu(fake_input)

    # 运行predictor
    predictor.zero_copy_run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])
    output_data = output_tensor.copy_to_cpu() # numpy.ndarray类型


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")

    return parser.parse_args()


def set_config(args):
    config = AnalysisConfig(args.model_file, args.params_file)
    config.disable_gpu()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    return config


if __name__ == "__main__":
    main()  
```
