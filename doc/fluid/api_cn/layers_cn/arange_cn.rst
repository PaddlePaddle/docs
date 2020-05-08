.. _cn_api_paddle_tensor_arange

arange
-------------------------------

.. py:function:: paddle.tensor.arange(start, end, step=1, dtype=None, name=None)

该API根据step均匀分隔给定数值区间[start, end)，并返回该分隔结果。

**参数**：
        - **start** （float32 | float64 | int32 | int64 | Variable） - 区间起点，且区间包括此值, 当类型是Variable时，是shape为 [1] 的1-D Tensor。
        - **end** （float32 | float64 | int32 | int64 | Variable） - 区间终点，通常区间不包括此值。但当step不是整数，且浮点数取整会影响输出的长度时例外。
        - **step** （float32 | float64 | int32 | int64 | Variable） - 均匀分割的步长。
        - **dtype** （str | core.VarDesc.VarType） - 输出Tensor的数据类型，可为 'float32', 'float64', 'int32', 'int64' 。

**返回**：均匀分割给定数值区间后得到的1-D Tensor, 数据类型为输入 dtype 。

**返回类型**：Variable

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    with fluid.dygraph.guard():
                 x = paddle.arange(0, 6, 2) 
                 # x: [0, 2, 4]
                 # x dtype: float32
