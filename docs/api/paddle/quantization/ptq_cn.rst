.. _cn_api_paddle_quantization_ptq:

PTQ
-------------------------------
.. py:class:: paddle.quantization.PTQ(Quantization)
将训练后量化应用到模型上。

方法
::::::::::::
quantize(model: Layer, inplace=False)
'''''''''

创建一个用于训练后量化的模型。

量化配置将在模型中传播。它将向模型中插入观察者以收集和计算量化参数。

**参数**
    - **model**(Layer) - 待量化的模型。
    - **inplace**(bool) - 是否对模型进行原地修改

**返回**
为训练后量化准备好的模型。

**代码示例**

COPY-FROM: paddle.quantization.PTQ.quantize

convert(model: paddle.nn.layer.layers.Layer, inplace=False, remain_weight=False)
'''''''''

将量化模型转换为ONNX格式。转换后的模型可以通过调用 paddle.jit.save 保存为推理模型。
参数 model：类型 model：Layer参数 inplace：类型 inplace：bool，可选参数 remain_weight：类型 remain_weight：bool，可选

返回
::::::::::
转换后的模型
