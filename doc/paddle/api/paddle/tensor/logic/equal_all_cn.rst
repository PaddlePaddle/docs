.. _cn_api_tensor_equal_all:

equal_all
-------------------------------

.. py:function:: paddle.equal_all(x, y, name=None)


该OP返回：返回的结果只有一个元素值，如果所有相同位置的元素相同返回True，否则返回False。

**注：该OP输出的结果不返回梯度。**


参数：
    - **x** (Tensor) - 输入Tensor，支持的数据类型包括 float32， float64，int32， int64。
    - **y** (Tensor) - 输入Tensor，支持的数据类型包括 float32， float64， int32， int64。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：输出结果为Tensor，Tensor数据类型为bool。

返回类型：变量（Tensor）

**代码示例**:

.. code-block:: python

     import numpy as np
     import paddle
     import paddle.imperative as imperative
     paddle.enable_imperative()
     paddle.enable_imperative()
     x = imperative.to_variable(np.array([1, 2, 3]))
     y = imperative.to_variable(np.array([1, 2, 3]))
     z = imperative.to_variable(np.array([1, 4, 3]))
     result1 = paddle.equal_all(x, y)
     print(result1.numpy()) # result1 = [True ]
     result2 = paddle.equal_all(x, z)
     print(result2.numpy()) # result2 = [False ]
