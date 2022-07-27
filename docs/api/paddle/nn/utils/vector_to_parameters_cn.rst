.. _cn_api_nn_utils_vector_to_parameters:

vector_to_parameters
-------------------------------

.. py:function:: paddle.nn.utils.vector_to_parameters(vec, parameters, name=None)

将1个1-D Tensor按顺序切分给输入的多个parameter。

参数
:::::::::
    - vec (Tensor) - 一个1-D Tensor。
    - parameters (Iterable[Tensor]) - 可迭代的多个parameter。parameter为Layer中可训练的Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
无

代码示例
:::::::::

COPY-FROM: paddle.nn.utils.vector_to_parameters
