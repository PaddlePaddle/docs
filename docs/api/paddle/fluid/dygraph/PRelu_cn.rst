.. _cn_api_fluid_dygraph_PRelu:

PRelu
-------------------------------

.. py:class:: paddle.fluid.dygraph.PRelu(mode, input_shape=None, param_attr=None, dtype="float32")




该接口用于构建 ``PRelu`` 类的一个可调用对象，具体用法参照 ``代码示例``。其中实现了 ``PRelu`` 激活函数的三种激活方式。

计算公式如下：

.. math::
    y = max(0, x) + \alpha min(0, x)


参数
::::::::::::

    - **mode** (str) - 权重共享模式。共提供三种激活方式：

    .. code-block:: text

        all：所有元素使用同一个alpha值
        channel：在同一个通道中的元素使用同一个alpha值
        element：每一个元素有一个独立的alpha值

    - **channel** (int，可选) - 通道数。该参数在mode参数为"channel"时是必须的。默认为None。
    - **input_shape** (int 或 list 或 tuple，可选) - 输入的维度。该参数在mode参数为"element"时是必须的。默认为None。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **dtype** (str，可选) - 数据类型，可以为"float32"或"float64"。默认值："float32"。

返回
::::::::::::
无

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph.base import to_variable
    import numpy as np

    inp_np = np.ones([5, 200, 100, 100]).astype('float32')
    with fluid.dygraph.guard():
        inp_np = to_variable(inp_np)
        prelu0 = fluid.PRelu(
           mode='all',
           param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
        dy_rlt0 = prelu0(inp_np)
        prelu1 = fluid.PRelu(
           mode='channel',
           channel=200,
           param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
        dy_rlt1 = prelu1(inp_np)
        prelu2 = fluid.PRelu(
           mode='element',
           input_shape=inp_np.shape,
           param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
        dy_rlt2 = prelu2(inp_np)

属性
::::::::::::
属性
::::::::::::
weight
'''''''''

本层的可学习参数，类型为 ``Parameter``

