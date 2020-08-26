.. _cn_api_fluid_layers_bpr_loss:

bpr_loss
-------------------------------

.. py:function:: paddle.fluid.layers.bpr_loss(input, label, name=None)

:alias_main: paddle.nn.functional.bpr_loss
:alias: paddle.nn.functional.bpr_loss,paddle.nn.functional.loss.bpr_loss
:old_api: paddle.fluid.layers.bpr_loss




贝叶斯个性化排序损失函数（Bayesian Personalized Ranking Loss Operator ）

该OP属于pairwise类型的损失函数。损失值由下式计算而得:

.. math::

  Y[i] = 1/(N[i] - 1) * \sum_j{\log(\sigma(X[i, Label[i]]-X[i, j]))}

其中：
    - :math:`X` ：输入值，一个形为[T x D]的2-D Tensor, 此处为logit值。
    - :math:`N[i]` ： 在时间步i的正例和负例的总和。
    - :math:`Label[i]` ：在时间步i的正例下标。
    - :math:`\sigma` ：激活函数。
    - :math:`Y` ：输出值，一个形为[T x 1]的2-D Tensor。
    

更多细节请参考 `Session Based Recommendations with Recurrent Neural Networks`

参数:
  - **input** (Variable) - 形为[T x D] , Tensor类型时T为batch大小，LoDTensor类型时T为mini-batch的总时间步。D 为正例加负例的个数。该输入为logits而非概率。数据类型是float32或float64。
  - **label** (Variable) - 形为[T x 1]，表示input中正例的下标，数据类型为int64。。
  - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回: 形为[T x 1]的2D张量，数据类型同input相同，表示bpr损失值。

返回类型：Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
     
    neg_size = 3
    # label=[0]
    label = fluid.layers.data(
              name="label", shape=[1], dtype="int64")
    # predict = [0.1, 0.2, 0.3, 0.4]
    predict = fluid.layers.data(
              name="predict", shape=[neg_size + 1], dtype="float32")
    # bpr_Loss : label [0] 表示predict中下标0表示正例，即为0.1, 负例有3个为0.2,0.3,0.4
    cost = fluid.layers.bpr_loss(input=predict, label=label)

