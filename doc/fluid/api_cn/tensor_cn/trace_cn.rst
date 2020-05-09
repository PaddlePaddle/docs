.. _cn_api_tensor_trace:

trace
-------------------------------

.. py:function:: fluid.layers.trace(input, offset=0, dim1=0, dim2=1)

该 OP 计算输入 Tensor 在指定平面上的对角线元素之和，并输出相应的计算结果。

如果输入是 2D Tensor，则返回对角线元素之和。 

如果输入的维度大于 2D，则返回一个由对角线元素之和组成的数组，其中对角线从由 dim1 和 dim2 指定的二维平面中获得。默认由输入的前两维组成获得对角线的 2D 平面。

参数 ``offset`` 确定从指定的二维平面中获取对角线的位置：

    - 如果 offset = 0，则取主对角线。
    - 如果 offset > 0，则取主对角线右上的对角线。
    - 如果 offset < 0，则取主对角线左下的对角线。

参数：
    - **input** （Variable）- 输入变量，至少为 2D 数组，支持数据类型为 float32，float64，int32，int64。
    - **offset** （int ，可选）- 从指定的二维平面中获取对角线的位置，默认值为 0，既主对角线。
    - **dim1** （int ， 可选）- 获取对角线的二维平面的第一维，默认值为 0。
    - **dim2** （int ， 可选）- 获取对角线的二维平面的第二维，默认值为 1。

返回： 指定二维平面的对角线元素之和。数据类型和输入数据类型一致。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.dygraph as dg
    import numpy as np
    
    case1 = np.random.randn(2, 3).astype('float32')
    case2 = np.random.randn(3, 10, 10).astype('float32')
    case3 = np.random.randn(3, 10, 5, 10).astype('float32')
    
    with dg.guard():
        case1 = dg.to_variable(case1)
        case2 = dg.to_variable(case2)
        case3 = dg.to_variable(case3)
        data1 = fluid.layers.trace(case1) # data1.shape = [1]
        data2 = fluid.layers.trace(case2, offset=1, dim1=1, dim2=2) # data2.shape = [3]
        data3 = fluid.layers.trace(case3, offset=-3, dim1=1, dim2=-1) # data2.shape = [3, 5]
