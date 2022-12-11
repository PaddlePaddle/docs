.. _cn_api_fluid_layers_bpr_loss:

bpr_loss
-------------------------------

.. py:function:: paddle.fluid.layers.bpr_loss(input, label, name=None)





贝叶斯个性化排序损失函数（Bayesian Personalized Ranking Loss Operator ）

该 OP 属于 pairwise 类型的损失函数。损失值由下式计算而得：

.. math::

  Y[i] = 1/(N[i] - 1) * \sum_j{\log(\sigma(X[i, Label[i]]-X[i, j]))}

其中：
    - :math:`X`：输入值，一个形为[T x D]的 2-D Tensor，此处为 logit 值。
    - :math:`N[i]`：在时间步 i 的正例和负例的总和。
    - :math:`Label[i]`：在时间步 i 的正例下标。
    - :math:`\sigma`：激活函数。
    - :math:`Y`：输出值，一个形为[T x 1]的 2-D Tensor。


更多细节请参考 `Session Based Recommendations with Recurrent Neural Networks`

参数
::::::::::::

  - **input** (Variable) - 形为[T x D] , Tensor 类型时 T 为 batch 大小，LoDTensor 类型时 T 为 mini-batch 的总时间步。D 为正例加负例的个数。该输入为 logits 而非概率。数据类型是 float32 或 float64。
  - **label** (Variable) - 形为[T x 1]，表示 input 中正例的下标，数据类型为 int64。。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 形为[T x 1]的 2DTensor，数据类型同 input 相同，表示 bpr 损失值。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.bpr_loss
