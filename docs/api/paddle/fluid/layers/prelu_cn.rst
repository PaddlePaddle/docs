.. _cn_api_fluid_layers_prelu:

prelu
-------------------------------

.. py:function:: paddle.static.nn.prelu(x, mode, param_attr=None, name=None)

prelu激活函数

.. math::
    prelu(x) = max(0, x) + \alpha * min(0, x)

共提供三种激活方式：

.. code-block:: text

    all: 所有元素使用同一个alpha值
    channel: 在同一个通道中的元素使用同一个alpha值
    element: 每一个元素有一个独立的alpha值


参数：
    - **x** （Tensor）- 多维Tensor或LoDTensor，数据类型为float32。
    - **mode** (str) - 权重共享模式。
    - **param_attr** (ParamAttr，可选) - 可学习权重 :math:`[\alpha]` 的参数属性，可由ParamAttr创建。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。 


返回： 表示激活输出Tensor，数据类型和形状于输入相同。

**代码示例：**

.. code-block:: python

    import paddle

    x = paddle.to_tensor([-1., 2., 3.])
    param = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.2))
    out = paddle.static.nn.prelu(x, 'all', param)
    # [-0.2, 2., 3.]


