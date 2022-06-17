.. _cn_api_fluid_dygraph_GroupNorm:

GroupNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.GroupNorm(channels, groups, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, data_layout='NCHW', dtype="float32")




**Group Normalization层**

该接口用于构建 ``GroupNorm`` 类的一个可调用对象，具体用法参照 ``代码示例``。其中实现了组归一化层的功能。更多详情请参考：`Group Normalization <https://arxiv.org/abs/1803.08494>`_ 。

参数
::::::::::::

    - **channels** (int) - 输入的通道数。
    - **groups** (int) - 从通道中分离出来的 ``group`` 的数目。
    - **epsilon** (float，可选) - 为防止方差除零，增加一个很小的值。默认值：1e-05。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **act** (str，可选) - 应用于输出上的激活函数，如tanh、softmax、sigmoid，relu等，支持列表请参考 :ref:`api_guide_activations`，默认值为None。
    - **data_layout** (str，可选) - 只支持“NCHW”(num_batches，channels，height，width)格式。默认值：“NCHW”。
    - **dtype** (str，可选) - 数据类型，可以为"float32"或"float64"。默认值为"float32"。

返回
::::::::::::
无

抛出异常
::::::::::::

    - ValueError - 如果 ``data_layout`` 不是“NCHW”格式。

代码示例
::::::::::::

..  code-block:: python

    import paddle.fluid as fluid
    import numpy s np

    with fluid.dygraph.guard():
        x = np.random.random((8, 32, 32)).astype('float32')
        groupNorm = fluid.dygraph.nn.GroupNorm(channels=32, groups=4)
        ret = groupNorm(fluid.dygraph.base.to_variable(x))


