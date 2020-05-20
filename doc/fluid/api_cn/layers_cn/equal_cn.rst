.. _cn_api_fluid_layers_equal:

equal
-------------------------------

.. py:function:: paddle.fluid.layers.equal(x,y,cond=None)




该OP返回 :math:`x==y` 逐元素比较x和y是否相等，x和y的维度应该相同。

参数：
    - **x** (Variable) - 输入Tensor，支持的数据类型包括 float32， float64，int32， int64。
    - **y** (Variable) - 输入Tensor，支持的数据类型包括 float32， float64， int32， int64。
    - **cond** (Variable，可选) - 逐元素比较的结果Tensor，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。

返回：输出结果的Tensor，输出Tensor的shape和输入一致，Tensor数据类型为bool。

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    out_cond = paddle.data(name='input1', shape=[2], dtype='bool')
    label = paddle.nn.functional.assign(np.array([3, 3], dtype='int32'))
    limit = paddle.nn.functional.assign(np.array([3, 2], dtype='int32'))
    label_cond = paddle.nn.functional.assign(np.array([1, 2], dtype='int32'))
    
    out1 = paddle.elementwise_equal(x=label, y=limit, name=None)
    out2 = paddle.elementwise_equal(x=label_cond, y=limit, name=None)

