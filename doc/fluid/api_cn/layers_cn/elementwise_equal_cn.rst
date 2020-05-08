.. _cn_api_tensor_elementwise_equal:

elementwise_equal
-------------------------------

.. py:function:: paddle.elementwise_equal(x, y, name=None)

该OP返回 :math:`x==y` 逐元素比较x和y是否相等。

参数：
    - **x** (Variable) - 输入Tensor，支持的数据类型包括 float32， float64，int32， int64。
    - **y** (Variable) - 输入Tensor，支持的数据类型包括 float32， float64， int32， int64。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：输出结果的Tensor，输出Tensor的shape和输入一致，Tensor数据类型为bool。

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    label = fluid.layers.assign(np.array([3, 3], dtype="int32"))
    limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
    out1 = paddle.elementwise_equal(x=label, y=limit) #out1=[True, False]
