.. _cn_api_fluid_layers_prelu:

prelu
-------------------------------

.. py:function:: paddle.fluid.layers.prelu(x, mode, param_attr=None, name=None)

等式：

.. math::
    y = max(0, x) + \alpha min(0, x)

共提供三种激活方式：

.. code-block:: text

    all: 所有元素使用同一个alpha值
    channel: 在同一个通道中的元素使用同一个alpha值
    element: 每一个元素有一个独立的alpha值


参数：
          - **x** （Variable）- 输入为Tensor。
          - **mode** (string) - 权重共享模式。
          - **param_attr** (ParamAttr|None) - 可学习权重 :math:`[\alpha]` 的参数属性，可由ParamAttr创建。
          - **name** （str | None）- 这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回： 输出Tensor与输入shape相同。

返回类型：  变量（Variable）

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.param_attr import ParamAttr
    x = fluid.layers.data(name="x", shape=[5,10,10], dtype="float32")
    mode = 'channel'
    output = fluid.layers.prelu(
             x,mode,param_attr=ParamAttr(name='alpha'))




