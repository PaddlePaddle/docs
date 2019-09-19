.. _cn_api_fluid_layers_bpr_loss:

bpr_loss
-------------------------------

.. py:function:: paddle.fluid.layers.bpr_loss(input, label, name=None)


贝叶斯个性化排序损失函数（Bayesian Personalized Ranking Loss Operator ）

该算子属于pairwise类型的损失函数。损失值由下式计算而得:

.. math::

  Y[i] = 1/(N[i] - 1) * \sum_j{\log(\sigma(X[i, Label[i]]-X[i, j]))}

其中：
    - :math:`X` ：输入值，一个形为[T x D]的2-D tensor, 此处为logit值。
    - :math:`N[i]` ： 在时间步i的正例和负例的总和。
    - :math:`Label[i]` ：在时间步i的正例下标。
    - :math:`\sigma` ：激活函数。
    - :math:`Y` ：输出值，一个形为[T x 1]的2-D tensor。
    

更多细节请参考 `Session Based Recommendations with Recurrent Neural Networks`

参数:
  - **input** (Tensor|LodTensor) - 数据类型为float32或者double。形为[T x D] , tensor类型时T为batch大小，LodTensor类型时T为mini-batch的总时间步。D 为正例加负例的个数。该输入为logits而非概率。
  - **label** (Tensor|LodTensor) - 数据类型为int64。形为[N x 1]，表示input中正例的下标。
  - **name** (str|None) - （可选）该层的命名。 如果为None, 则自动为该层命名。 默认为None。

返回: 形为[T x 1]的2D张量，数据类型同input相同，表示bpr损失值。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
     
    neg_size = 10
    label = fluid.layers.data(
              name="label", shape=[1], dtype="int64")
    predict = fluid.layers.data(
              name="predict", shape=[neg_size + 1], dtype="float32")
    cost = fluid.layers.bpr_loss(input=predict, label=label)

