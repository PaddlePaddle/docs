# 使用 X2Paddle 迁移推理模型

X2Paddle 是飞桨生态下的模型转换工具，致力于帮助你快速迁移其他深度学习框架至飞桨框架。目前支持**推理模型的框架转换**与**PyTorch训练代码迁移**，除此之外还提供了详细的不同框架间API对比文档，降低你上手飞桨核心的学习成本。

## 迁移 PyTorch、ONNX、TensorFlow 以及 Caffe 模型

### 具体用法

#### PyTorch 模型转换

```python
from x2paddle.convert import pytorch2paddle
pytorch2paddle(module=torch_module,
               save_dir="./pd_model",
               jit_type="trace",
               input_examples=[torch_input])
# module (torch.nn.Module): PyTorch的Module。
# save_dir (str): 转换后模型的保存路径。
# jit_type (str): 转换方式。默认为"trace"。
# input_examples (list[torch.tensor]): torch.nn.Module的输入示例，list的长度必须与输入的长度一致。默认为None。
```

` `  ` script `  ` ` 模式以及更多细节可参考[PyTorch模型转换文档](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/pytorch2paddle.md)。

#### TensorFlow 模型转换

```shell
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model
```

#### ONNX 模型转换

```shell
x2paddle --framework=onnx --model=onnx_model.onnx --save_dir=pd_model
```

#### Caffe 模型转换

```shell
x2paddle --framework=caffe --prototxt=deploy.prototxt --weight=deploy.caffemodel --save_dir=pd_model
```

#### 转换参数说明

| 参数                 | 作用                                                         |
| -------------------- | ------------------------------------------------------------ |
| --framework          | 源模型类型 (tensorflow、caffe、onnx)                         |
| --prototxt           | 当framework为caffe时，该参数指定caffe模型的proto文件路径     |
| --weight             | 当framework为caffe时，该参数指定caffe模型的参数文件路径      |
| --save_dir           | 指定转换后的模型保存目录路径                                 |
| --model              | 当framework为tensorflow/onnx时，该参数指定tensorflow的pb模型文件或onnx模型路径 |
| --caffe_proto        | **[可选]** 由caffe.proto编译成caffe_pb2.py文件的存放路径，当存在自定义Layer时使用，默认为None |
| --define_input_shape | **[可选]** For TensorFlow, 当指定该参数时，强制用户输入每个Placeholder的shape，见[文档](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/FAQ.md) |
| --enable_code_optim  | **[可选]** For PyTorch, 是否对生成代码进行优化，默认为True |
| --to_lite            | **[可选]** 是否使用opt工具转成Paddle-Lite支持格式，默认为False |
| --lite_valid_places  | **[可选]** 指定转换类型，可以同时指定多个backend(以逗号分隔)，opt将会自动选择最佳方式，默认为arm |
| --lite_model_type    | **[可选]** 指定模型转化类型，目前支持两种类型：protobuf和naive_buffer，默认为naive_buffer |
| --disable_feedback   | **[可选]** 是否关闭X2Paddle使用反馈；X2Paddle默认会统计用户在进行模型转换时的成功率，以及转换框架来源等信息，以便于帮忙X2Paddle根据用户需求进行迭代，不会上传用户的模型文件。如若不想参与反馈，可指定此参数为False即可 |

#### X2Paddle API

目前X2Paddle提供API方式转换模型，可参考[X2PaddleAPI](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/x2paddle_api.md)

#### 一键转换Paddle-Lite支持格式

可参考[使用X2paddle导出Padde-Lite支持格式](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/convert2lite_api.md)

## 迁移其他框架模型

如果您是其他框架，例如 MXNet、MindSpore，先导出 ONNX，再通过下列命令，将 ONNX 模型转为 Paddle 模型

```shell
x2paddle --framework=onnx --model=onnx_model.onnx --save_dir=pd_model
```

***【注意】*** 欢迎大家前往GitHub给[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)点击Star，关注项目，即可随时了解X2Paddle的最新进展。
