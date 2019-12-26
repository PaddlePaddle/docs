.. _cn_api_fluid_layers_equal:

equal
-------------------------------

.. py:function:: paddle.fluid.layers.equal(x,y,cond=None)

该OP返回 :math:`x==y` 逐元素比较x和y是否相等，x和y的维度应该相同。

参数：
    - **x** (Variable) - 输入Tensor，支持的数据类型包括 float32， float64，int32， int64。
    - **y** (Variable) - 输入Tensor，支持的数据类型包括 float32， float64， int32， int64。
    - **cond** (Variable，可选) - 逐元素比较的结果Tensor，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。
    - **force_cpu** (bool，可选) – 是否强制将输出Tensor存储在CPU。默认值为None，表示将输出Tensor存储在CPU内存上；如果为False，则将输出Tensor存储在运行设备内存上。

返回：输出结果的Tensor，输出Tensor的shape和输入一致，Tensor数据类型为bool。

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    out_cond =fluid.data(name="input1", shape=[2], dtype='bool')
    label = fluid.layers.assign(np.array([3, 3], dtype="int32"))
    limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
    out0 = fluid.layers.equal(x=label,y=limit) #out1=[True, False]
    out1 = fluid.layers.equal(x=label,y=limit, cond=out_cond) #out2=[True, False] out_cond=[True, False]
    out2 = fluid.layers.equal(x=label,y=limit,force_cpu=False) #out3=[True, False]
    out3 = label == limit # out3=[True, False]


