.. _cn_api_nn_utils_vector_to_parameters:

vector_to_parameters
-------------------------------

.. py:function:: paddle.nn.utils.vector_to_parameters(vec, parameters, name=None)

将 1 个 1-D Tensor 按顺序切分给输入的多个 parameter。

参数
:::::::::
    - **vec** (Tensor) - 一个 1-D Tensor，它将被切片并复制到输入参数(input parameters)中。
    - **parameters** (Iterable[Tensor]) - 可迭代的多个 parameter。parameter 为 Layer 中可训练的 Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
无

代码示例
:::::::::

COPY-FROM: paddle.nn.utils.vector_to_parameters
