.. _cn_api_fluid_dygraph_GroupNorm:

GroupNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.GroupNorm(name_scope, groups, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, data_layout='NCHW')

**Group Normalization层**

该OP实现了组归一化层。更多详情请参考： `Group Normalization <https://arxiv.org/abs/1803.08494>`_ 。

参数：
    - **name_scope** (str) - 该类的名称。
    - **groups** (int) - 从通道中分离出来的 ``group`` 的数目。
    - **epsilon** (float, 可选) - 为防止方差除零，增加一个很小的值。默认值：1e-05。
    - **param_attr** (ParamAttr, 可选) - 可学习标度 :math:`g` 的参数属性，如果设置为False，则不会向输出单元添加标度。如果设置为None，标度初始化为1。默认值：None。
    - **bias_attr** (ParamAttr, 可选) - 可学习偏置 :math:`b ` 的参数属性，如果设置为False，则不会向输出单元添加偏置量。如果设置为None，偏置初始化为0。默认值：None。
    - **act** (str, 可选) - 指明激活函数类型。默认值：None。
    - **data_layout** (str, 可选) - 只支持"NCHW"(num_batches，channels，height，width)格式。默认值："NCHW"。

返回：无

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy s np

    with fluid.dygraph.guard():
        x = np.random.random((8, 32, 32)).astype('float32')
        groupNorm = fluid.dygraph.nn.GroupNorm('GroupNorm', groups=4)
        ret = groupNorm(fluid.dygraph.base.to_variable(x))

