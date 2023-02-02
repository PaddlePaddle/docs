.. _cn_api_nn_cn_log_sigmoid:

log_sigmoid
-------------------------------

.. py:function:: paddle.nn.functional.log_sigmoid(x, name=None)

log_sigmoid 激活层。计算公式如下：

.. math::

    log\_sigmoid(x) = \log \frac{1}{1 + e^{-x}}

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.log_sigmoid
