.. _cn_api_fluid_layers_reverse:

reverse
-------------------------------

.. py:function:: paddle.fluid.layers.reverse(x,axis)

:alias_main: paddle.reverse
:alias: paddle.reverse,paddle.tensor.reverse,paddle.tensor.manipulation.reverse
:old_api: paddle.fluid.layers.reverse



**reverse**

该OP对输入Tensor ``x`` 在指定轴 ``axis`` 上进行数据的逆序操作。

::

    示例1:
        输入是 LoDTensor 类型:
            x = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            axis = [0, 1]

        输出:
            output = [[8, 7, 6], [5, 4, 3], [2, 1, 0]]

    示例2:
        输入是 LoDTensorArray 类型:
            x = {[[0, 1], [2, 3]],
                 [[4, 5, 6]],
                 [[7], [8], [9]]}
            axis = 0

        输出:
            output = {[[7], [8], [9]],
                      [[4, 5, 6]],
                      [[0, 1], [2, 3]]}

参数：
  - **x** (Variable) - 输入为Tensor或LoDTensorArray，数据类型支持bool，int8，int32，int64，float32和float64。若输入是LoDTensorArray类型，则返回一个逆序的LoDTensorArray，其内部Tensor元素的次序保持不变。
  - **axis** (int|tuple|list) - 指定逆序运算的轴，取值范围是[-R, R)，R是输入 ``x`` 的Rank， ``axis`` 为负时与 ``axis`` +R 等价。如果 ``axis`` 是一个元组或列表，则在 ``axis`` 每个元素值所指定的轴上进行逆序运算。如果输入是LoDTensorArray类型，axis须是值为0的int，或shape为[1]的list ``[0]`` 、元组 ``(0,)`` 。
返回：逆序后的Tensor，形状、数据类型和 ``x`` 一致。

返回类型：Variable

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        data = fluid.layers.assign(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype='float32')) # [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]
        result1 = fluid.layers.reverse(data, 0) # [[6., 7., 8.], [3., 4., 5.], [0., 1., 2.]]
        result2 = fluid.layers.reverse(data, [0, 1]) # [[8., 7., 6.], [5., 4., 3.], [2., 1., 0.]]

        # 输入为LoDTensorArray时
        data1 = fluid.layers.assign(np.array([[0, 1, 2]], dtype='float32'))
        data2 = fluid.layers.assign(np.array([[3, 4, 5]], dtype='float32'))
        tensor_array = fluid.layers.create_array(dtype='float32')
        i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        fluid.layers.array_write(data1, i, tensor_array)
        fluid.layers.array_write(data2, i+1, tensor_array)

        reversed_tensor_array = fluid.layers.reverse(tensor_array, 0) # {[[3, 4, 5]], [[0, 1, 2]]}
