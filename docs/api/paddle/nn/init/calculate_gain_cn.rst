.. _cn_api_paddle_nn_init_calculate_gain:

calculate_gain
-------------------------------

.. py:class:: paddle.nn.init.calculate_gain(nonlinearity, param=None)

返回给定非线性函数的推荐增益值。

参数
::::::::::::

    - **nonlinearity** (str) - 非线形函数的名称。
    - **param** (float，可选) - 某些非线性函数的可选参数。目前，该参数仅适用于'leaky_relu'。默认值：无（在公式中会自动计算为 0.01）。

返回
::::::::::::

    float 类型增益值。
