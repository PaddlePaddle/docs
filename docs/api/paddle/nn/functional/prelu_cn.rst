.. _cn_api_paddle_nn_functional_prelu:

prelu
-------------------------------

.. py:function:: paddle.nn.functional.prelu(x, weight, data_format="NCHW", name=None)

prelu 激活。计算公式如下：

.. math::

    prelu(x) = max(0, x) + weight * min(0, x)

其中，:math:`x` 和 ``weight`` 为输入的 Tensor

参数
::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。别名 ``input``。
    - **weight** (Tensor) - 可训练参数，数据类型同 ``x`` 一致，形状支持：[]、[1] 或者 [in]，其中 ``in`` 为输入 ``x`` 的通道数。
    - **name** (str|None，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **data_format** (str，可选) - 指定输入的数据格式，可以是 "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" 或 "NDHWC"。默认值："NCHW"。

返回
:::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.prelu
