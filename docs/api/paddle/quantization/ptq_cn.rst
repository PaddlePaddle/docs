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

convert(self, model:layer, inplace=False, remain_weight=False):
'''''''''

将量化模型转换为ONNX格式。转换后的模型可以通过调用 paddle.jit.save 保存为推理模型。

**参数**
    - **model**(Layer) - 待量化的模型。
    - **inplace**(bool, optional) - 是否要对模型进行就地修改，默认为false。
    - **remain_weight**(bool, optional) - 是否宝石权重为floats，默认为false。

**返回**
转换后的模型
