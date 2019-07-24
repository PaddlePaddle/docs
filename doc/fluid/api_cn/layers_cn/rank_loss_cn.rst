.. _cn_api_fluid_layers_rank_loss:

rank_loss
-------------------------------


.. py:function::  paddle.fluid.layers.rank_loss(label, left, right, name=None)

`RankNet <http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf>`_ 是一个成对的
排序模型，训练样本由一对文档组成：A和B。标签P表示a的排名是否高于B:

P 的取值可为： {0, 1} 或 {0, 0.5, 1}, 其中，0.5表示输入的两文档排序相同。

排序的损失函数有三个输入:left(o_i)、right(o_j) 和 label (P\_{i,j})。输入分别表示RankNet对文档A、B的输出得分和标签p的值。由下式计算输入的排序损失C\_{i,j}:

.. math::

   C_{i,j} &= -\tilde{P_{ij}} * o_{i,j} + \log(1 + e^{o_{i,j}}) \\
      o_{i,j} &=  o_i - o_j  \\
      \tilde{P_{i,j}} &= \left \{0, 0.5, 1 \right \} \ or \ \left \{0, 1 \right \}

排序损失层的输入带有batch_size (batch_size >= 1)

参数：
  - **label** (Variable)：A的排名是否高于B
  - **left** (Variable)：RankNet对doc A的输出分数
  - **right** (Variable)：RankNet对doc B的输出分数
  - **name** (str|None)：此层的名称(可选)。如果没有设置，层将自动命名。

返回：rank loss的值

返回类型： list

抛出异常： ``ValueError`` - label, left, 和right至少有一者不是variable变量类型。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name="label", shape=[-1, 1], dtype="float32")
    left = fluid.layers.data(name="left", shape=[-1, 1], dtype="float32")
    right = fluid.layers.data(name="right", shape=[-1, 1], dtype="float32")
    out = fluid.layers.rank_loss(label, left, right)



