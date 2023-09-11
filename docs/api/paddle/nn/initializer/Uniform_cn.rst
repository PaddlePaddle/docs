.. _cn_api_paddle_nn_initializer_Uniform:

Uniform
-------------------------------

.. py:class:: paddle.nn.initializer.Uniform(low=-1.0, high=1.0, name=None)


均匀分布初始化方法。

参数
::::::::::::

    - **low** (float，可选) - 均匀分布的下界，默认值为 :math:`-1.0`。
    - **high** (float，可选) - 均匀分布的上界，默认值为 :math:`1.0`。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
由均匀分布初始化的参数。

代码示例
::::::::::::

COPY-FROM: paddle.nn.initializer.Uniform
