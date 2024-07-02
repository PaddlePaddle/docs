.. _cn_api_paddle_nn_FeatureAlphaDropout:

FeatureAlphaDropout
-------------------------------

.. py:function:: paddle.nn.FeatureAlphaDropout(p=0.5, name=None)

一个通道是一个特征图， `FeatureAlphaDropout` 会随机屏蔽掉整个通道。 `AlphaDropout` 是一种具有自归一化性质的 `dropout` 。均值为 0 ，方差为 1 的输入，经过 `AlphaDropout` 计算之后，输出的均值和方差与输入保持一致。 `AlphaDropout` 通常与 SELU 激活函数组合使用。论文请参考：`Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_

在动态图模式下，请使用模型的 `eval()` 方法切换至测试阶段。

.. note::
   对应的 `functional 方法` 请参考：:ref:`cn_api_paddle_nn_functional_feature_alpha_dropout` 。

参数
:::::::::
 - **p** (float)：将输入节点置 0 的概率，即丢弃概率。默认：0.5。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
`Tensor` ，经过 FeatureAlphaDropout 之后的结果，与输入 x 形状相同的 `Tensor` 。

代码示例
:::::::::

COPY-FROM: paddle.nn.FeatureAlphaDropout
