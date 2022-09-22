.. _cn_api_tensor_clone:

clone
-------------------------------

.. py:function:: paddle.clone(x, name=None)

对输入 Tensor :attr:`x` 进行拷贝，并返回一个新的 Tensor。

除此之外，该 API 提供梯度计算，在计算反向时，输出 Tensor 的梯度将会回传给输入 Tensor。

参数
:::::::::
    - **x** (Tensor) - 输入 Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

``Tensor``，从输入 :attr:`x` 拷贝的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.clone
