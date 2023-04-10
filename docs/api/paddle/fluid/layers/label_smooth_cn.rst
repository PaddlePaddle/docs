.. _cn_api_fluid_layers_label_smooth:

label_smooth
-------------------------------

.. py:function:: paddle.fluid.layers.label_smooth(label, prior_dist=None, epsilon=0.1, dtype='float32', name=None)




该OP实现了标签平滑的功能。标签平滑是一种对分类器层进行正则化的机制，称为标签平滑正则化(LSR)。由于直接优化正确标签的对数似然可能会导致过拟合，降低模型的适应能力，因此提出了标签平滑的方法来降低模型置信度。

标签平滑使用标签 :math:`y` 和一些固定模式随机分布变量 :math:`\mu`。对 :math:`k` 标签，标签平滑的计算方式如下。

.. math::

            \tilde{y_k} = (1 - \epsilon) * y_k + \epsilon * \mu_k,

其中 :math:`1-\epsilon` 和 :math:`\epsilon` 分别是权重，:math:`\tilde{y_k}` 是平滑后的标签，通常 :math:`\mu` 使用均匀分布。


关于更多标签平滑的细节，`查看论文  <https://arxiv.org/abs/1512.00567>`_ 。


参数
::::::::::::

  - **label** （Variable） - 包含标签数据的输入变量。标签数据应使用 one-hot 表示，是维度为 :math:`[N_1, ..., Depth]` 的多维Tensor，其中Depth为字典大小。
  - **prior_dist** （Variable，可选） - 用于平滑标签的先验分布，是维度为 :math:`[1，class\_num]` 的2D Tensor。如果未设置，则使用均匀分布。默认值为None。
  - **epsilon** （float，可选） - 用于混合原始真实分布和固定分布的权重。默认值为0.1。
  - **dtype** （np.dtype|core.VarDesc.VarType|str，可选） - 输入 ``Tensor`` 的数据类型，数据类型可以为“float16”，float32“或”float64“。默认值为”float32“。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
为平滑后标签的 ``Tensor``，数据类型为dtype设置的数据类型，维度也与输入的label参数维度相同。

返回类型
::::::::::::
 Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.label_smooth
