.. _cn_api_paddle_nn_init:

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

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.calculate_gain


uniform_
-------------------------------

.. py:class:: paddle.nn.init.uniform_(tensor, a=0.0, b=1.0)

将输入张量的值设置为均匀分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **a** (float) - 均匀分布的下限，默认值为 0.0。
    - **b** (float) - 均匀分布的上限，默认值为 1.0。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.uniform_


normal_
-------------------------------

.. py:class:: paddle.nn.init.normal_(tensor, mean=0.0, std=1.0)

将输入张量的值设置为正态分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **mean** (float) - 正态分布的均值，默认值为 0.0。
    - **std** (float) - 正态分布的标准差，默认值为 1.0。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.normal_


constant_
-------------------------------

.. py:class:: paddle.nn.init.constant_(tensor, val)

将输入张量的值设置为指定的常量，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **val** (float) - 指定的填充输入张量的常量。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.constant_


ones_
-------------------------------

.. py:class:: paddle.nn.init.ones_(tensor)

将输入张量的值设置为 1，该操作会直接修改输入张量。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.ones_


zeros_
-------------------------------

.. py:class:: paddle.nn.init.zeros_(tensor)

将输入张量的值设置为 0，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.zeros_


eye_
-------------------------------

.. py:class:: paddle.nn.init.eye_(tensor)

将二维输入张量的值设置为单位矩阵，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.eye_


dirac_
-------------------------------

.. py:class:: paddle.nn.init.dirac_(tensor, groups=1)

将输入张量的值设置为 Dirac delta 函数，该操作会直接修改输入张量。支持 3D、4D 和 5D 输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **groups** (int，可选) - 输入张量的组数，默认值为 1。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.dirac_


xavier_uniform_
-------------------------------

.. py:class:: paddle.nn.init.xavier_uniform_(tensor, gain=1.0)

将输入张量的值设置为 Xavier 均匀分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **gain** (float，可选) - 比例因子，默认值为 1.0。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.xavier_uniform_


xavier_normal_
-------------------------------

.. py:class:: paddle.nn.init.xavier_normal_(tensor, gain=1.0)

将输入张量的值设置为 Xavier 正常分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **gain** (float，可选) - 比例因子，默认值为 1.0。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.xavier_normal_


kaiming_uniform_
-------------------------------

.. py:class:: paddle.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

将输入张量的值设置为 Kaiming 均匀分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **a** (float，可选) - 该层之后使用的修正函数的负斜率（仅在使用'leaky_relu'时使用），默认值为 0。
    - **mode** (str，可选) - 计算增益的方法，可选值为'fan_in'、'fan_out'，默认值为'fan_in'。
    - **nonlinearity** (str，可选) - 非线性函数的名称，默认值为'leaky_relu'。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.kaiming_uniform_


kaiming_normal_
-------------------------------

.. py:class:: paddle.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

将输入张量的值设置为 Kaiming 正态分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **a** (float，可选) - 该层之后使用的修正函数的负斜率（仅在使用'leaky_relu'时使用），默认值为 0。
    - **mode** (str，可选) - 计算增益的方法，可选值为'fan_in'、'fan_out'，默认值为'fan_in'。
    - **nonlinearity** (str，可选) - 非线性函数的名称，默认值为'leaky_relu'。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.kaiming_normal_


trunc_normal_
-------------------------------

.. py:class:: paddle.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)

将输入张量的值设置为截断的正态分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **mean** (float，可选) - 正态分布的均值，默认值为 0.0。
    - **std** (float，可选) - 正态分布的标准差，默认值为 1.0。
    - **a** (float，可选) - 截断的下限，默认值为-2.0。
    - **b** (float，可选) - 截断的上限，默认值为 2.0。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.trunc_normal_


orthogonal_
-------------------------------

.. py:class:: paddle.nn.init.orthogonal_(tensor, gain=1)

将输入张量的值设置为截断的正态分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **gain** (float，可选) - 比例因子，默认值为 1.0。

代码示例
::::::::::::

COPY-FROM: paddle.nn.init.orthogonal_
