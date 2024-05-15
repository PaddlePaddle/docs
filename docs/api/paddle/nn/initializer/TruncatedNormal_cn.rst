.. _cn_api_paddle_nn_initializer_TruncatedNormal:

TruncatedNormal
-------------------------------

.. py:class:: paddle.nn.initializer.TruncatedNormal(mean=0.0, std=1.0, a=-2.0, b=2.0, name=None)

.. note::
    在参数设置时建议将 `mean` 设为 :math:`a \le mean \le b`。
    若 :math:`mean < a - 2 \cdot std` 或 :math:`mean > b + 2 \cdot std`，采样值的分布可能是有误的。

截断正态分布（高斯分布）初始化方法。

参数
    - **mean** (float，可选) - 正态分布的均值，默认值为 :math:`0.0`。
    - **std** (float，可选) - 正态分布的标准差，默认值为 :math:`1.0`。
    - **a** (float，可选) - 截断正态分布的下界，默认值为 :math:`-2.0`。
    - **b** (float，可选) - 截断正态分布的上界，默认值为 :math:`2.0`。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    由截断正态分布（高斯分布）初始化的参数。

代码示例
::::::::::::

COPY-FROM: paddle.nn.initializer.TruncatedNormal
