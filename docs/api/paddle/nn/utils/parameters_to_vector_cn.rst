.. _cn_api_nn_utils_parameters_to_vector:

parameters_to_vector
-------------------------------

.. py:function:: paddle.nn.utils.parameters_to_vector(parameters, name=None)

将输入的多个 parameter 展平并连接为 1 个 1-D Tensor。

参数
:::::::::
    - **parameters** (Iterable[Tensor]) - 可迭代的多个 parameter。parameter 为 Layer 中可训练的 Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，多个 parameter 展平并连接的 1-D Tensor

代码示例
:::::::::

COPY-FROM: paddle.nn.utils.parameters_to_vector
