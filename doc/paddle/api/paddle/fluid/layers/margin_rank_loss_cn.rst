.. _cn_api_fluid_layers_margin_rank_loss:

margin_rank_loss
-------------------------------

.. py:function:: paddle.fluid.layers.margin_rank_loss(label, left, right, margin=0.1, name=None)




margin rank loss（间隔排序损失）层。在排序问题中，它可以比较来自排序网络的输入 ``left`` 和输入 ``right`` 的得分。

可用如下等式定义：

.. math::
    rank\_loss = max(0, -label * (left - right) + margin)


参数:
  - **label** (Variable) – 表示输入 ``left`` 的真实排序是否高于输入 ``right`` , 数据类型为 float32。
  - **left** (Variable) – 输入 ``left`` 的排序得分， 数据类型为 float32 。
  - **right** (Variable) – 输入 ``right`` 的排序得分， 数据类型为 float32。
  - **margin** (float) – 指定的间隔。
  - **name** (str，可选) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值为None。

返回： 排序损失

返回类型: Variable

抛出异常:
  - ``ValueError`` - ``label`` , ``left`` , ``right`` 有一者不为Variable类型时，抛出此异常

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name="label", shape=[-1, 1], dtype="float32")
    left = fluid.layers.data(name="left", shape=[-1, 1], dtype="float32")
    right = fluid.layers.data(name="right", shape=[-1, 1], dtype="float32")
    out = fluid.layers.margin_rank_loss(label, left, right)











