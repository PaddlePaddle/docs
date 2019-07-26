.. _cn_api_fluid_layers_margin_rank_loss:

margin_rank_loss
-------------------------------

.. py:function:: paddle.fluid.layers.margin_rank_loss(label, left, right, margin=0.1, name=None)

margin rank loss（差距排序损失）层。在排序问题中，它可以比较传进来的 ``left`` 得分和 ``right`` 得分。

可用如下等式定义：

.. math::
    rank\_loss = max(0, -label * (left - right) + margin)


参数:
  - **label** (Variable) – 表明是否左元素排名要高于右元素
  - **left** (Variable) – 左元素排序得分
  - **right** (Variable) – 右元素排序得分
  - **margin** (float) – 指定固定的得分差
  - **name** (str|None) – 可选项，该层的命名。如果为None, 该层将会被自动命名

返回： 排序损失

返回类型: 变量（Variable）

抛出异常:
  - ``ValueError`` - ``label`` , ``left`` , ``right`` 有一者不为Variable类型时，抛出此异常

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name="label", shape=[-1, 1], dtype="float32")
    left = fluid.layers.data(name="left", shape=[-1, 1], dtype="float32")
    right = fluid.layers.data(name="right", shape=[-1, 1], dtype="float32")
    out = fluid.layers.margin_rank_loss(label, left, right)











