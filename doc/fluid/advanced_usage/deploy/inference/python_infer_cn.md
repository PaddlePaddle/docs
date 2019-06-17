# Python预测API介绍
Fluid提供了高度优化的C++预测库，为了方便使用，Python中也提供了调用C++预测引擎的API，下面是详细说明

## PaddleBuf
`PaddleBuf`定义了Tensor的存储结构，创建`PaddleBuf`:
``` python
int64_buf = PaddleBuf([1, 2, 3, 4])
float_buf = PaddleBuf([1., 2., 3., 4.])
```

`PadleBuf`包括以下方法
* `resize`: 重新分配内存，单位为byte
* `reset`: 重新设置数据
* `empty`: buffer是否为空
* `float\_data`: 将数据转为float型的list返回
* `int64\_data`: 将数据转为int64型的list返回
* `length`: 内存大小，单位为byte

## PaddleDType
`PaddleDType`定义了Tensor的类型，目前包括
* `PaddleDType.INT64`: 64位整型
* `PaddleDType.FLOAT32`: 32位浮点型

## PaddleTensor
`PaddleTensor`是预测库输入和输出的数据结构，包括以下字段
* `name`(str): 指定输入的名称
* `shape`(tuple|list): Tensor的shape
* `data`(PaddleBuf): Tensor的数据，存储在`PaddleBuf`中，
* `dtype`(PaddleDType): Tensor的类型

## AnalysisConfig
`AnalysisConfig`是创建预测引擎的配置，主要包括以下方法
* `set\_model`: 设置模型的路径
* `model\_dir`: 返回模型路径
* `enable\_use\_gpu`: 设置GPU显存(单位M)和ID
* `disable\_gpu`: 禁用GPU
* `gpu\_device\_id`: 返回使用的GPU ID
* `switch\_ir\_optim`: IR优化(默认开启)
* `enable\_tensorrt\_engine`: 启用TensorRT
* `enable\_mkldnn`: 启用MKLDNN


## 使用预测引擎
创建预测引擎的配置

``` python
# 创建预测引擎
config = fluid.core.AnalysisConfig(model_dir)
config.enable_use_gpu(200, 0) # 200M显存, 设备id为0
config.enable_tensorrt_engine() # 打开TensorRT

predictor = fluid.core.create_paddle_predictor(config)

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
```
