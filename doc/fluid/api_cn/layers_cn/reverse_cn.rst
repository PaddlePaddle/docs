.. _cn_api_fluid_layers_reverse:

reverse
-------------------------------

.. py:function:: paddle.fluid.layers.reverse(x,axis)

**reverse**

该OP对输入Tensor ``x`` 在指定轴 ``axis`` 上进行数据的逆序操作。

参数
::::::::::::

  - **x** (Variable) - 多维Tensor，类型必须为int32，int64，float32，float64。
  - **axis** (int|tuple|list) - 指定逆序运算的轴，取值范围是[-R, R)，R是输入 ``x`` 的Rank， ``axis`` 为负时与 ``axis`` +R 等价。如果 ``axis`` 是一个元组或列表，则在``axis`` 每个元素值所指定的轴上进行逆序运算。

返回
::::::::::::
逆序后的Tensor，形状、数据类型和 ``x`` 一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

.. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        data = fluid.layers.assign(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype='float32')) # [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]
        result1 = fluid.layers.reverse(data, 0) # [[6., 7., 8.], [3., 4., 5.], [0., 1., 2.]]
        result2 = fluid.layers.reverse(data, [0, 1]) # [[8., 7., 6.], [5., 4., 3.], [2., 1., 0.]]
