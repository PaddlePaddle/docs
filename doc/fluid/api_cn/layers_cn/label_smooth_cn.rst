.. _cn_api_fluid_layers_label_smooth:

label_smooth
-------------------------------

.. py:function:: paddle.fluid.layers.label_smooth(label, prior_dist=None, epsilon=0.1, dtype='float32', name=None)

标签平滑是一种对分类器层进行正则化的机制，称为标签平滑正则化(LSR)。


由于直接优化正确标签的对数似然可能会导致过拟合，降低模型的适应能力，因此提出了标签平滑的方法来降低模型置信度。
标签平滑使用标签 :math:`y` 自身和一些固定模式随机分布变量 :math:`\mu` 。对 :math:`k` 标签，我们有：

.. math::
            \tilde{y_k} = (1 - \epsilon) * y_k + \epsilon * \mu_k,

其中 :math:`1-\epsilon` 和 :math:`\epsilon` 分别是权重， :math:`\tilde{y_k}` 是平滑后的标签。 通常μ 使用均匀分布


查看更多关于标签平滑的细节 https://arxiv.org/abs/1512.00567

参数：
  - **label** （Variable） - 包含标签数据的输入变量。 标签数据应使用 one-hot 表示。
  - **prior_dist** （Variable） - 用于平滑标签的先验分布。 如果未提供，则使用均匀分布。 prior_dist的shape应为 :math:`(1，class\_num)` 。
  - **epsilon** （float） - 用于混合原始真实分布和固定分布的权重。
  - **dtype** （np.dtype | core.VarDesc.VarType | str） - 数据类型：float32，float_64，int等。
  - **name** （str | None） - 此层的名称（可选）。 如果设置为None，则将自动命名图层。

返回：张量变量, 包含平滑后的标签

返回类型: Variable

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers

    label = fluid.layers.data(name="label", shape=[1], dtype="float32")
    one_hot_label = fluid.layers.one_hot(input=label, depth=10)
    smooth_label = fluid.layers.label_smooth(
    label=one_hot_label, epsilon=0.1, dtype="float32")









