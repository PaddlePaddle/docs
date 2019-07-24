.. _cn_api_fluid_layers_range:

range
-------------------------------

.. py:function:: paddle.fluid.layers.range(start, end, step, dtype)

均匀分隔给定数值区间，并返回该分隔结果。

返回值在半开区间[start，stop)内生成（即包括起点start但不包括终点stop的区间）。


参数：
    - **start** （int | float | Variable） - 区间起点，且区间包括此值。
    - **end** （int | float | Variable） - 区间终点，通常区间不包括此值。但当step不是整数，且浮点数取整会影响out的长度时例外。
    - **step** （int | float | Variable） - 返回结果中数值之间的间距（步长）。 对于任何输出变量out，step是两个相邻值之间的距离，即out [i + 1]  -  out [i]。 默认为1。
    - **dtype** （string） - 'float32'|'int32'| ...，输出张量的数据类型。

返回：均匀分割给定数值区间后得到的值组


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.range(0, 10, 2, 'int32')





