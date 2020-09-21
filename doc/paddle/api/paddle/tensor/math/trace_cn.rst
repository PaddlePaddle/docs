.. _cn_api_tensor_trace:

trace
-------------------------------

.. py:function:: paddle.trace(x, offset=0, axis1=0, axis2=1, name=None)




该 OP 计算输入 Tensor 在指定平面上的对角线元素之和，并输出相应的计算结果。

如果输入是 2D Tensor，则返回对角线元素之和。 

如果输入的维度大于 2D，则返回一个由对角线元素之和组成的数组，其中对角线从由 axis1 和 axis2 指定的二维平面中获得。默认由输入的前两维组成获得对角线的 2D 平面。

参数 ``offset`` 确定从指定的二维平面中获取对角线的位置：

    - 如果 offset = 0，则取主对角线。
    - 如果 offset > 0，则取主对角线右上的对角线。
    - 如果 offset < 0，则取主对角线左下的对角线。

参数：
    - **x** （Variable）- 输入张量，至少为 2D 数组，支持数据类型为 float32，float64，int32，int64。
    - **offset** （int ，可选）- 从指定的二维平面中获取对角线的位置，默认值为 0，既主对角线。
    - **axis1** （int ， 可选）- 获取对角线的二维平面的第一维，默认值为 0。
    - **axis2** （int ， 可选）- 获取对角线的二维平面的第二维，默认值为 1。
    - **name** (str，可选)- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： 指定二维平面的对角线元素之和。数据类型和输入数据类型一致。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python

    import paddle
    import numpy as np
    
    case1 = np.random.randn(2, 3).astype('float32')
    case2 = np.random.randn(3, 10, 10).astype('float32')
    case3 = np.random.randn(3, 10, 5, 10).astype('float32')
    
    paddle.enable_imperative()
    case1 = paddle.imperative.to_variable(case1)
    case2 = paddle.imperative.to_variable(case2)
    case3 = paddle.imperative.to_variable(case3)
    data1 = paddle.trace(case1) # data1.shape = [1]
    data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) # data2.shape = [3]
    data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) # data2.shape = [3, 5]
