.. _cn_api_tensor_not_equal:

not_equal
-------------------------------
.. py:function:: paddle.not_equal(x, y, name=None)

:alias_main: paddle.not_equal
:alias: paddle.not_equal,paddle.tensor.not_equal,paddle.tensor.logic.not_equal

该OP返回 :math:`x!=y` 逐元素比较x和y是否相等，相同位置的元素不相同则返回True，否则返回False。使用重载算子 `!=` 可以有相同的计算函数效果

**注：该OP输出的结果不返回梯度。**

参数：
    - **x** (Tensor) - 输入Tensor，支持的数据类型包括 float32， float64，int32， int64。
    - **y** (Tensor) - 输入Tensor，支持的数据类型包括 float32， float64， int32， int64。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    

返回：输出结果的Tensor，输出Tensor的shape和输入一致，Tensor数据类型为bool。

返回类型：变量（Tensor）

**代码示例**:

.. code-block:: python

     import numpy as np
     import paddle
     import paddle.imperative as imperative
     paddle.enable_imperative()
     x = imperative.to_variable(np.array([1, 2, 3]))
     y = imperative.to_variable(np.array([1, 3, 2]))
     result1 = paddle.not_equal(x, y)
     print(result1.numpy())  # result1 = [False True True]


