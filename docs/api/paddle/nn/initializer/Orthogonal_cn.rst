.. _cn_api_nn_initializer_Orthogonal:

Orthogonal
-------------------------------

.. py:class:: paddle.nn.initializer.Orthogonal(gain=1.0, name=None)

正交矩阵初始化，被初始化的参数为 (半)正交的。

该初始化策略仅适用于 2-D及以上的参数。对于维度超过2的参数，将0维作为行数 ，将1维及之后的维度展平为列数。

具体可以描述为：

.. code-block:: text

    rows = shape[0]
    cols = shape[1]·shape[1]···shape[N]

    if rows < cols:
        The rows are orthogonal vectors
    elif rows > cols:
        The columns are orthogonal vectors
    else rows = cols:
        Both rows and columns are orthogonal vectors

参数
:::::::::
    - gain (float，可选) - 参数初始化的增益系数，可通过 :ref:`cn_api_nn_initializer_calculate_gain` 获取推荐的增益系数。默认：1.0
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回：参数初始化类的实例

代码示例
:::::::::

.. code-block:: python

    import paddle

    weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal())
    linear = paddle.nn.Linear(10, 15, weight_attr=weight_attr)
    # linear.weight: X * X' = I

    linear = paddle.nn.Linear(15, 10, weight_attr=weight_attr)
    # linear.weight: X' * X = I
    