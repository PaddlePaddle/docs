.. _cn_api_fluid_dygraph_GroupNorm:

GroupNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.GroupNorm(name_scope, groups, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, data_layout='NCHW')

**Group Normalization层**

请参考 `Group Normalization <https://arxiv.org/abs/1803.08494>`_ 。

参数：
    - **name_scope** (str) - 该类名称
    - **groups** (int) - 从 channel 中分离出来的 group 的数目
    - **epsilon** (float) - 为防止方差除零，增加一个很小的值
    - **param_attr** (ParamAttr|None)  - 可学习标度的参数属性 :math:`g`,如果设置为False，则不会向输出单元添加标度。如果设置为0，偏差初始化为1。默认值:None
    - **bias_attr** (ParamAttr|None) - 可学习偏置的参数属性 :math:`b ` , 如果设置为False，则不会向输出单元添加偏置量。如果设置为零，偏置初始化为零。默认值:None。
    - **act** (str) - 将激活应用于输出的 group normalizaiton
    - **data_layout** (string|NCHW) - 只支持NCHW。

返回： 一个张量变量，它是对输入进行 group normalization 后的结果。

返回类型：Variable


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        x = numpy.random.random((8, 32, 32)).astype('float32')
        groupNorm = fluid.dygraph.nn.GroupNorm('GroupNorm', groups=4)
        ret = groupNorm(fluid.dygraph.base.to_variable(x))






