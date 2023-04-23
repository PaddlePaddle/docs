.. _cn_api_nn_initializer_Orthogonal:

Orthogonal
-------------------------------

.. py:class:: paddle.nn.initializer.Orthogonal(gain=1.0, name=None)

正交矩阵初始化，被初始化的参数为 (半)正交的。

该初始化策略仅适用于 2-D 及以上的参数。对于维度超过 2 的参数，将 0 维作为行数，将 1 维及之后的维度展平为列数。

具体可以描述为：

.. code-block:: text

    rows = shape[0]
    cols = shape[1]·shape[2]···shape[N]

    if rows < cols:
        The rows are orthogonal vectors
    elif rows > cols:
        The columns are orthogonal vectors
    else rows = cols:
        Both rows and columns are orthogonal vectors

参数
:::::::::
    - **gain** (float，可选) - 参数初始化的增益系数，可通过 :ref:`cn_api_nn_initializer_calculate_gain` 获取推荐的增益系数。默认：1.0
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
该参数初始化的类实例对象

代码示例
:::::::::

COPY-FROM: paddle.nn.initializer.Orthogonal
