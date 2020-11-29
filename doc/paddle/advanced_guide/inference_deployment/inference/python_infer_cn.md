# Python 预测 API介绍

Paddle提供了高度优化的[C++预测库](./native_infer.html)，为了方便使用，我们也提供了C++预测库对应的Python接口，下面是详细的使用说明。

如果您在使用2.0之前的Paddle，请参考[旧版API](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/advanced_guide/inference_deployment/inference/python_infer_cn.html)文档。

## Python预测相关数据结构

使用Python预测API与C++预测API相似，主要包括`Tensor`, `DataType`, `Config`和`Predictor`，分别对应于C++ API中同名的类型。

### DataType

class paddle.inference.DataType

`DataType`定义了`Tensor`的数据类型，由传入`Tensor`的numpy数组类型确定，包括以下成员

* `INT64`: 64位整型
* `INT32`: 32位整型
* `FLOAT32`: 32位浮点型

### PrecisionType

class paddle.3.inference.PrecisionType

`PrecisionType`定义了`Predictor`运行的精度模式，包括一下成员

* `Float32`: fp32模式运行
* `Half`: fp16模式运行
* `Int8`: int8模式运行

### Tensor

class paddle.inference.Tensor

`Tensor`是`Predictor`的一种输入/输出数据结构，通过`predictor`获取输入/输出handle得到，主要提供以下方法

* `copy_from_cpu`: 从cpu获取模型运行所需输入数据
* `copy_to_cpu`: 获取模型运行输出结果
* `lod`: 获取lod信息
* `set_lod`: 设置lod信息
* `shape`: 获取shape信息
* `reshape`: 设置shape信息
* `type`: 获取DataType信息

``` python
# 创建predictor
predictor = create_predictor(config)

# 获取输入的名称
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_handle(input_names[0])

# 设置输入
fake_input = numpy.random.randn(1, 3, 318, 318).astype("float32")
input_tensor.copy_from_cpu(fake_input)

# 运行predictor
predictor.run()

# 获取输出
output_names = predictor.get_output_names()
output_tensor = predictor.get_output_handle(output_names[0])
output_data = output_tensor.copy_to_cpu() # numpy.ndarray类型
```

### Config

class paddle.inference.Config

`Config`是创建预测引擎的配置，提供了模型路径设置、预测引擎运行设备选择以及多种优化预测流程的选项，主要包括以下方法

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
config = Config("./model")
```
* 当模型文件夹下只有一个模型文件和一个参数文件时，传入模型文件和参数文件路径
``` python
config = Config("./model/model", "./model/params")
```
使用`set_model`方法设置模型和参数路径方式同上

其他预测引擎配置选项示例如下
``` python
config.enable_use_gpu(100, 0) # 初始化100M显存，使用gpu id为0
config.gpu_device_id()        # 返回正在使用的gpu id
config.disable_gpu()          # 禁用gpu
config.switch_ir_optim(True)  # 开启IR优化
config.enable_tensorrt_engine(precision_mode=PrecisionType.Float32,
                              use_calib_mode=True) # 开启TensorRT预测，精度为fp32，开启int8离线量化
config.enable_mkldnn()          # 开启MKLDNN
```

### Predictor

class paddle.inference.Predictor

`Predictor`是运行预测的引擎，由`paddle.inference.create_predictor(config)`创建，主要提供以下方法

* `run()`: 运行预测引擎，返回预测结果
* `get_input_names()`: 获取输入的名称
* `get_input_handle(input_name: str)`: 根据输入的名称获取对应的`Tensor`
* `get_output_names()`: 获取输出的名称
* `get_output_handle(output_name: str)`: 根据输出的名称获取对应的`Tensor`

#### 代码示例

``` python
# 设置完AnalysisConfig后创建预测引擎PaddlePredictor
predictor = create_predictor(config)

# 获取输入的名称
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])

# 设置输入
fake_input = numpy.random.randn(1, 3, 318, 318).astype("float32")
input_handle.reshape([1, 3, 318, 318])
input_handle.copy_from_cpu(fake_input)

# 运行predictor
predictor.run()

# 获取输出
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
```



## 完整使用示例

下面是使用Paddle Inference Python API进行预测的一个完整示例，使用resnet50模型

下载[resnet50模型](http://paddle-inference-dist.bj.bcebos.com/resnet50_model.tar.gz)并解压，运行如下命令将会调用预测引擎

``` bash
python resnet50_infer.py --model_file ./model/model --params_file ./model/params --batch_size 2
```

`resnet50_infer.py` 的内容是

``` python
import argparse
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor


def main():
    args = parse_args()

    # 设置AnalysisConfig
    config = set_config(args)

    # 创建PaddlePredictor
    predictor = create_predictor(config)

    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 设置输入
    fake_input = np.random.randn(1, 3, 318, 318).astype("float32")
    input_handle.reshape([1, 3, 318, 318])
    input_handle.copy_from_cpu(fake_input)

    # 运行predictor
    predictor.run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu() # numpy.ndarray类型


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")

    return parser.parse_args()


def set_config(args):
    config = Config(args.model_file, args.params_file)
    config.disable_gpu()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    return config


if __name__ == "__main__":
    main()
```

## 支持方法列表

* Tensor
    * `copy_from_cpu(input: numpy.ndarray) -> None`
    * `copy_to_cpu() -> numpy.ndarray`
    * `reshape(input: numpy.ndarray|List[int]) -> None`
    * `shape() -> List[int]`
    * `set_lod(input: numpy.ndarray|List[List[int]]) -> None`
    * `lod() -> List[List[int]]`
    * `type() -> PaddleDType`
* Config
    * `set_model(model_dir: str) -> None`
    * `set_model(prog_file: str, params_file: str) -> None`
    * `set_model_buffer(model: str, model_size: int, param: str, param_size: int) -> None`
    * `model_dir() -> str`
    * `prog_file() -> str`
    * `params_file() -> str`
    * `model_from_memory() -> bool`
    * `set_cpu_math_library_num_threads(num: int) -> None`
    * `enable_use_gpu(memory_pool_init_size_mb: int, device_id: int) -> None`
    * `use_gpu() -> bool`
    * `gpu_device_id() -> int`
    * `switch_ir_optim(x: bool = True) -> None`
    * `switch_ir_debug(x: int=True) -> None`
    * `ir_optim() -> bool`
    * `enable_tensorrt_engine(workspace_size: int = 1 << 20,
                              max_batch_size: int,
                              min_subgraph_size: int,
                              precision_mode: AnalysisConfig.precision,
                              use_static: bool,
                              use_calib_mode: bool) -> None`
    * `set_trt_dynamic_shape_info(min_input_shape: Dict[str, List[int]]={}, max_input_shape: Dict[str, List[int]]={}, optim_input_shape: Dict[str, List[int]]={}, disable_trt_plugin_fp16: bool=False) -> None`
    * `tensorrt_engine_enabled() -> bool`
    * `enable_mkldnn() -> None`
    * `enable_mkldnn_bfloat16() -> None`
    * `mkldnn_enabled() -> bool`
    * `set_mkldnn_cache_capacity(capacity: int=0) -> None`
    * `set_mkldnn_op(ops: Set[str]) -> None`
    * `set_optim_cache_dir(dir: str) -> None`
    * `disable_glog_info() -> None`
    * `pass_builder() -> paddle::PassStrategy`
    * `delete_pass(pass_name: str) -> None`
    * `cpu_math_library_num_threads() -> int`
    * `disable_gpu() -> None`
    * `enable_lite_engine(precision: PrecisionType, zero_copy: bool, passes_filter: List[str]=[], ops_filter: List[str]=[]) -> None`
    * `lite_engine_enabled() -> bool`
    * `enable_memory_optim() -> None`
    * `enable_profile() -> None`
    * `enable_quantizer() -> None`
    * `quantizer_config() -> paddle::MkldnnQuantizerConfig`
    * `fraction_of_gpu_memory_for_pool() -> float`
    * `memory_pool_init_size_mb() -> int`
    * `glog_info_disabled() -> bool`
    * `gpu_device_id() -> int`
    * `specify_input_name() -> bool`
    * `switch_specify_input_names(x: bool=True) -> None`
    * `specify_input_name(q) -> bool`
    * `switch_use_feed_fetch_ops(x: int=True) -> None`
    * `use_feed_fetch_ops_enabled() -> bool`
    * `to_native_config() -> paddle.fluid.core_avx.NativeConfig`
* `create_predictor(config: Config) -> Predictor`
* Predictor
    * `run() -> None`
    * `get_input_names() -> List[str]`
    * `get_input_handle(input_name: str) -> Tensor`
    * `get_output_names() -> List[str]`
    * `get_output_handle(output_name: str) -> Tensor`
    * `clear_intermediate_tensor() -> None`
    * `try_shrink_memory() -> None`
    * `clone() -> Predictor`
* PredictorPool
    * `retrive(idx: int) -> Predictor`

可参考对应的[C++预测接口](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/inference_api.cc)，其中定义了每个接口的参数和返回值
