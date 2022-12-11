.. _cn_api_fluid_layers_range:

range
-------------------------------

.. py:function:: paddle.fluid.layers.range(start, end, step, dtype, name=None)


注意：推荐使用 paddle.arange

该OP返回以步长 ``step`` 均匀分隔给定数值区间[``start``, ``end``)的1-D Tensor，数据类型为 ``dtype``。

当 ``dtype`` 表示浮点类型时，为了避免浮点计算误差，建议给 ``end`` 加上一个极小值epsilon，使边界可以更加明确。

参数
::::::::::::

        - **start** (float|int|Tensor) - 区间起点（且区间包括此值）。当 ``start`` 类型是Tensor时，是形状为[1]且数据类型为int32、int64、float32、float64的Tensor。
        - **end** (float|int|Tensor) - 区间终点（且通常区间不包括此值）。当 ``end`` 类型是Tensor时，是形状为[1]且数据类型为int32、int64、float32、float64的Tensor。
        - **step** (float|int|Tensor) - 均匀分割的步长。当 ``step`` 类型是Tensor时，是形状为[1]且数据类型为int32、int64、float32、float64的Tensor。
        - **dtype** (str|np.dtype|core.VarDesc.VarType) - 输出Tensor的数据类型，支持int32、int64、float32、float64。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

        Tensor：以步长 ``step`` 均匀分割给定数值区间[``start``, ``end``)后得到的1-D Tensor，数据类型为 ``dtype`` 。

抛出异常
::::::::::::

        - ``TypeError`` - 如果 ``dtype`` 不是int32、int64、float32、float64。

代码示例：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.range(0, 10, 2, 'int32')
    # [0, 2, 4, 6, 8]
