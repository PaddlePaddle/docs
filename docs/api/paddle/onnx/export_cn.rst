.. _cn_api_paddle_onnx_export:

export
-----------------

.. py:function:: paddle.onnx.export(layer, path, input_spec=None, opset_version=9, **configs)

将输入的 ``Layer`` 存储为 ``ONNX`` 格式的模型，可使用 onnxruntime 或其他框架进行推理。

.. note::

    具体使用案例请参考 :ref:`cn_model_to_onnx`，更多信息请参考：`paddle2onnx <https://github.com/PaddlePaddle/paddle2onnx>`_ 。

参数
:::::::::
    - **layer** (Layer) - 导出的 ``Layer`` 对象。
    - **path** (str) - 存储模型的路径前缀。格式为 ``dirname/file_prefix`` 或者 ``file_prefix``，导出后``ONNX``模型自动添加后缀 ``.onnx`` 。
    - **input_spec** (list[InputSpec|Tensor]，可选) - 描述存储模型 forward 方法的输入，可以通过 InputSpec 或者示例 Tensor 进行描述。如果为 ``None``，所有原 ``Layer`` forward 方法的输入变量将都会被配置为存储模型的输入变量。默认为 ``None``。
    - **opset_version** (int，可选) - 导出 ``ONNX`` 模型的 Opset 版本，目前稳定支持导出的版本为 9、10 和 11。默认为 ``9``。
    - **configs (dict，可选) - 其他用于兼容的存储配置选项。这些选项将来可能被移除，如果不是必须使用，不推荐使用这些配置选项。默认为 ``None``。目前支持以下配置选项：(1) output_spec (list[Tensor]) - 选择存储模型的输出目标。默认情况下，所有原 ``Layer`` forward 方法的返回值均会作为存储模型的输出。如果传入的 ``output_spec`` 列表不是所有的输出变量，存储的模型将会根据 ``output_spec`` 所包含的结果被裁剪。

返回
:::::::::
无。

代码示例
:::::::::

COPY-FROM: paddle.onnx.export
