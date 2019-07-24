.. _cn_api_fluid_layers_bpr_loss:

bpr_loss
-------------------------------

.. py:function:: paddle.fluid.layers.bpr_loss(input, label, name=None)


贝叶斯个性化排序损失计算（Bayesian Personalized Ranking Loss Operator ）

该算子属于pairwise的排序类型，其标签是期望物品。在某次会话中某一给定点的损失值由下式计算而得:

.. math::

  \[Y[i] = 1/(N[i] - 1) * \sum_j{\log(\sigma(X[i, Label[i]]-X[i, j]))}\]

更多细节请参考 `Session Based Recommendations with Recurrent Neural Networks`_

参数:
  - **input** (Variable|list) - 一个形为[N x D]的2-D tensor , 其中 N 为批大小batch size ，D 为种类的数量。该输入为logits而非概率。
  - **label** (Variable|list) - 2-D tensor<int64> 类型的真实值, 形为[N x 1]
  - **name** (str|None) - （可选）该层的命名。 如果为None, 则自动为该层命名。 默认为None.

返回: 形为[N x 1]的2D张量，即bpr损失

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
     
    neg_size = 10
    label = fluid.layers.data(
              name="label", shape=[1], dtype="int64")
    predict = fluid.layers.data(
              name="predict", shape=[neg_size + 1], dtype="float32")
    cost = fluid.layers.bpr_loss(input=predict, label=label)





