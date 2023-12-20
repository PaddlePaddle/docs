.. _cn_api_paddle_quantization_qat:

qat
-------------------------------

.. py:class:: paddle.quantization.QAT(config: paddle.quantization.config.QuantConfig)
用于为量化感知训练准备模型的工具。

参数
:::::::::
    - **config** (QuantConfig) - 量化配置,通常指的是设置和调整模型量化过程中的参数和选项。

代码示例
::::::::::

COPY-FROM: paddle.quantization.qat

.. py:class:: quantize(model: paddle.nn.layer.layers.Layer, inplace=False)
创建一个适用于量化感知训练的模型。

量化配置将在模型中传播。并且它将在模型中插入伪量化器以模拟量化过程。

参数
:::::::::
    - **model(Layer)**-
    - **inplace(tool)**-

返回
:::::::::
为量化感知训练准备好的模型。

代码示例
::::::::::

COPY-FROM: paddle.quantization.qat
        
.. py:function:: convert(model: paddle.nn.layer.layers.Layer, inplace=False, remain_weight=False)
将量化模型转换为ONNX格式。转换后的模型可以通过调用paddle.jit.save保存为推理模型。:参数模型::类型模型: Layer:原地操作参数:
:原理类型: bool, optional:剩余权重参数::类型剩余权重: bool, optional

返回
:::::::::
转换后的模型

代码示例
::::::::::

>>> import paddle
>>> from paddle.quantization import QAT, QuantConfig
>>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
>>> from paddle.vision.models import LeNet

>>> quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
>>> q_config = QuantConfig(activation=quanter, weight=quanter)
>>> qat = QAT(q_config)
>>> model = LeNet()
>>> quantized_model = qat.quantize(model)
>>> converted_model = qat.convert(quantized_model)
>>> dummy_data = paddle.rand([1, 1, 32, 32], dtype="float32")
>>> paddle.jit.save(converted_model, "./quant_deploy", [dummy_data])
>>> dummy_data = paddle.rand([1, 1, 32, 32], dtype="float32")
>>> paddle.jit.save(converted_model, "./quant_deploy", [dummy_data])
