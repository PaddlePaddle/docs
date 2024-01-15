.. _cn_api_paddle_quantization_QAT:

QAT
-------------------------------

.. py:class:: paddle.quantization.QAT(config: paddle.quantization.config.QuantConfig)
用于为量化感知训练准备模型的工具。

参数
::::::::::::
    - **config** (QuantConfig) - 量化配置,通常指的是设置和调整模型量化过程中的参数和选项。

**代码示例**

COPY-FROM: paddle.quantization.QAT.quantize

方法
::::::::::::
quantize(model: Layer, inplace=False)
'''''''''
创建一个适用于量化感知训练的模型。

量化配置将在模型中传播。并且它将在模型中插入伪量化器以模拟量化过程。

**参数**
    - **model(Layer)** - 待量化的模型
    - **inplace(bool)** - 是否对模型进行原地修改

**返回**
为量化感知训练准备好的模型。

**代码示例**

COPY-FROM: paddle.quantization.QAT.quantize
        
方法
::::::::::::
convert(model: paddle.nn.layer.layers.Layer, inplace=False, remain_weight=False)
'''''''''
将量化模型转换为ONNX格式。转换后的模型可以通过调用paddle.jit.save保存为推理模型。:参数模型::类型模型: Layer:原地操作参数:
:原理类型: bool, optional:剩余权重参数::类型剩余权重: bool, optional

**返回**
转换后的模型
