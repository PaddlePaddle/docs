.. _cn_api_paddle_nn_initializer_Normal:

Normal
-------------------------------

.. py:class:: paddle.nn.initializer.Normal(mean=0.0, std=1.0, name=None)


随机正态（高斯）分布初始化函数。

参数
::::::::::::

    - **mean** (float，可选) - 正态分布的平均值。默认值为 0。
    - **std** (float，可选) - 正态分布的标准差。默认值为 1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    由随机正态（高斯）分布初始化的参数。

代码示例
::::::::::::

COPY-FROM: paddle.nn.initializer.Normal
