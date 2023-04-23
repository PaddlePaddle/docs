.. _cn_api_fluid_layers_instance_norm:

instance_norm
-------------------------------


.. py:function:: paddle.static.nn.instance_norm(input, epsilon=1e-05, param_attr=None, bias_attr=None, name=None)





可用作卷积和全连接操作的实例正则化函数，根据每个样本的每个通道的均值和方差信息进行正则化。该层需要的数据格式如下：

NCHW[batch,in_channels,in_height,in_width]

更多详情请参考：`Instance Normalization: The Missing Ingredient for Fast Stylization <https://arxiv.org/pdf/1607.08022.pdf>`_

``input`` 是 mini-batch 的输入。

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mean of each channel in each sample in a batch  \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// variance of each channel in each sample a batch  \\
    \hat{x_i}          &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta                                                      \quad &// scale-and-shift


参数
::::::::::::

    - **input** (Tensor) - instance_norm 算子的输入特征，是一个 Tensor，输入的维度可以为 2, 3, 4, 5。数据类型：float32 和 float64。
    - **epsilon** （float，默认 1e-05）-为了当前输入做标准化时得到稳定的结果而加在的分母上的扰动值。默认值为 1e-5。
    - **param_attr** （ParamAttr|None） - instance_norm 权重参数的属性，可以设置为 None 或者一个 ParamAttr 的类（ParamAttr 中可以指定参数的各种属性）。如果设为 None，则默认的参数初始化为 1.0。如果在 ParamAttr 指定了属性时，instance_norm 创建相应属性的 param_attr（权重）参数。默认：None。
    - **bias_attr** （ParamAttr|None） - instance_norm 偏置参数的属性，可以设置为 None 或者一个 ParamAttr 的类（ParamAttr 中可以指定参数的各种属性）。如果设为 None，默认的参数初始化为 0.0。如果在 ParamAttr 指定了参数的属性时，instance_norm 创建相应属性的 bias_attr（偏置）参数。默认：None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，在输入中运用 instance normalization 后的结果。


代码示例
::::::::::::

COPY-FROM: paddle.static.nn.instance_norm
