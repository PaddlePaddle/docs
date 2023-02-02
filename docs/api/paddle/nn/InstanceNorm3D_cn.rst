.. _cn_api_nn_InstanceNorm3D:

InstanceNorm3D
-------------------------------

.. py:class:: paddle.nn.InstanceNorm3D(num_features, epsilon=1e-05, momentum=0.9, weight_attr=None, bias_attr=None, data_format="NCDHW", name=None)

构建 ``InstanceNorm3D`` 类的一个可调用对象，具体用法参照 ``代码示例``。可以处理 5D 的 Tensor，实现了实例归一化层（Instance Normalization Layer）的功能。更多详情请参考：Instance Normalization: The Missing Ingredient for Fast Stylization 。

数据布局：NCDHW [batch, in_channels, D, in_height, in_width]

``input`` 是 mini-batch 的输入。

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mean  \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// variance \\
    \hat{x_i}          &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta                                                      \quad &// scale-and-shift

其中 `H` 是高度，`W` 是宽度。


参数
::::::::::::

    - **num_features** (int) - 指明输入 ``Tensor`` 的通道数量。
    - **epsilon** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **momentum** (float，可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var``。默认值：0.9。更新公式如上所示。
    - **weight_attr** (ParamAttr|bool，可选) - 指定权重参数属性的对象。如果为 False，则表示每个通道的伸缩固定为 1，不可改变。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。如果为 False，则表示每一个通道的偏移固定为 0，不可改变。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **data_format** (string，可选) - 指定输入数据格式，数据格式可以为"NCDHW"。默认值：“NCDHW”。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。


形状
::::::::::::

    - input：形状为 5-D Tensor。
    - output：和输入形状一样。

.. note::
目前设置 track_running_stats 和 momentum 是无效的。之后的版本会修复此问题。


代码示例
::::::::::::

COPY-FROM: paddle.nn.InstanceNorm3D
