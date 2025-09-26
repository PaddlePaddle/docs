.. _cn_api_paddle_nn_init_kaiming_uniform_:

kaiming_uniform\_
-------------------------------

.. py:class:: paddle.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

将输入张量的值设置为 Kaiming 均匀分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **a** (float，可选) - 该层之后使用的修正函数的负斜率（仅在使用'leaky_relu'时使用），默认值为 0。
    - **mode** (str，可选) - 计算增益的方法，可选值为'fan_in'、'fan_out'，默认值为'fan_in'。
    - **nonlinearity** (str，可选) - 非线性函数的名称，默认值为'leaky_relu'。
