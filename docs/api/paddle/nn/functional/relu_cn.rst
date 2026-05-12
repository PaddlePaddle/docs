.. _cn_api_paddle_nn_functional_relu:

relu
-------------------------------

.. py:function:: paddle.nn.functional.relu(x, inplace=False, name=None)

relu 激活层（Rectified Linear Unit）。计算公式如下：

.. math::

    relu(x) = max(0, x)

其中，:math:`x` 为输入的 Tensor。


参数
::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。别名 ``input``。
    - **inplace** (bool，可选) - 是否使用 inplace 操作。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.relu
