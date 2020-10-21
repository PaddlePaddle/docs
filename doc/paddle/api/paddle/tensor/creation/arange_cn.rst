.. _cn_api_paddle_tensor_arange

arange
-------------------------------

.. py:function:: paddle.arange(start=0, end=None, step=1, dtype=None, name=None)




该OP返回以步长 ``step`` 均匀分隔给定数值区间[``start``, ``end``)的1-D Tensor，数据类型为 ``dtype``。

当 ``dtype`` 表示浮点类型时，为了避免浮点计算误差，建议给 ``end`` 加上一个极小值epsilon，使边界可以更加明确。

参数
::::::::::
        - **start** (float|int|Tensor) - 区间起点（且区间包括此值）。当 ``start`` 类型是Tensor时，是形状为[1]且数据类型为int32、int64、float32、float64的Tensor。如果仅指定 ``start`` ，而 ``end`` 为None，则区间为[0, ``start``)。默认值为0。
        - **end** (float|int|Tensor, 可选) - 区间终点（且通常区间不包括此值）。当 ``end`` 类型是Tensor时，是形状为[1]且数据类型为int32、int64、float32、float64的Tensor。默认值为None。
        - **step** (float|int|Tensor, 可选) - 均匀分割的步长。当 ``step`` 类型是Tensor时，是形状为[1]且数据类型为int32、int64、float32、float64的Tensor。默认值为1。
        - **dtype** (str|np.dtype|core.VarDesc.VarType, 可选) - 输出Tensor的数据类型，支持int32、int64、float32、float64。当该参数值为None时， 输出Tensor的数据类型为int64。默认值为None.
        - **name** (str, 可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回
::::::::::
        Tensor: 以步长 ``step`` 均匀分割给定数值区间[``start``, ``end``)后得到的1-D Tensor, 数据类型为 ``dtype`` 。

抛出异常
::::::::::
        - ``TypeError`` - 如果 ``dtype`` 不是int32、int64、float32、float64。

代码示例
::::::::::

.. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()

        out1 = paddle.arange(5)
        # [0, 1, 2, 3, 4]

        out2 = paddle.arange(3, 9, 2.0)
        # [3, 5, 7]

        # use 4.999 instead of 5.0 to avoid floating point rounding errors
        out3 = paddle.arange(4.999, dtype='float32')
        # [0., 1., 2., 3., 4.]

        start_var = paddle.imperative.to_variable(np.array([3]))
        out4 = paddle.arange(start_var, 7)
        # [3, 4, 5, 6]
