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
* `enable_use_gpu`: 设置GPU显存(单位M)和device id
* `disable_gpu`: 禁用GPU
* `gpu_device_id`: 返回使用的GPU ID
* `switch_ir_optim`: IR优化(默认开启)
* `enable_tensorrt_engine`: 启用TensorRT
* `enable_mkldnn`: 启用MKLDNN
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
config.enable_use_gpu(100, 0) # 初始化200M显存，使用gpu id为0
config.gpu_device_id()        # 返回正在使用的gpu id
config.disable_gpu()		  # 禁用gpu
config.switch_ir_optim(True)  # 开启IR优化 
config.enable_tensorrt_engine(precision=AnalysisConfig.Precision.kFloat32,
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

# 运行预测引擎得到结果，返回值是一个PaddleTensor的list
results = predictor.run([x_t, y_t])

# 获得预测结果，并应用到自己的应用中
```
## 支持方法列表
* PaddleTensor
	* `as_ndarray() -> numpy.ndarray`
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
                              precision: AnalysisConfig.precision,                                   				   use_static: bool, 
                              use_calib_mode: bool) -> None`
	* `enable_mkldnn() -> None`
* PaddlePredictor
	* `run(input: List[PaddleTensor]) -> List[PaddleTensor]`

可参考对应的[C++预测接口](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/inference_api.cc)，其中定义了每个接口的参数和返回值

## 完整使用示例

下面是使用Fluid Python API进行预测的一个完整示例，使用resnet50模型

下载[resnet50模型](http://paddle-inference-dist.bj.bcebos.com/resnet50_model.tar.gz)并解压，运行如下命令将会调用预测引擎

``` bash
python resnet50_infer.py --model_file ./model/model --params_file ./model/params --batch_size 2
```

`resnet50_infer.py` 的内容是

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