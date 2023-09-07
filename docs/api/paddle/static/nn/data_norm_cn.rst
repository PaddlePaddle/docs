.. _cn_api_paddle_static_nn_data_norm:

data_norm
-------------------------------

.. py:function:: paddle.static.nn.data_norm(input, act=None, epsilon=1e-05, param_attr=None, data_layout='NCHW', in_place=False, name=None, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, slot_dim=-1, sync_stats=False, summary_decay_rate=0.9999999, enable_scale_and_shift=False)




**数据正则化层**

可用作 conv2d 和 fully_connected 操作的正则化函数。此层所需的数据格式为以下之一：

1. NHWC [batch, in_height, in_width, in_channels]
2. NCHW [batch, in_channels, in_height, in_width]

:math:`input` 为一个 mini-batch 上的特征：

.. math::
        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &//\
        \ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \
        \mu_{\beta})^2 \qquad &//\ mini-batch\ variance \\
        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

参数
::::::::::::

  - **input** (Tensor) - 输入变量。
  - **act** (str，可选) - 激活函数类型，线性| relu | prelu | ...，默认值为 None。
  - **epsilon** (float，可选) - 指明在计算过程中是否添加较小的值到方差中以防止除零。默认值：1e-05。
  - **param_attr** (ParamAttr，可选) - 参数比例的参数属性。默认值为 None。
  - **data_layout** (str，可选) -  指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"。
  - **in_place** (bool，可选) - 是否使 data_norm 的输入和输出复用同一块内存，默认值为 False。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **moving_mean_name** (str，可选) - 存储全局 Mean 的 moving_mean 的名称。默认值为 None。
  - **moving_variance_name** (str，可选) - 存储全局 Variance 的 moving_variance 的名称。默认值为 None。
  - **do_model_average_for_mean_and_var** (bool，可选) - 是否为 mean 和 variance 进行模型平均。默认值为 False。
  - **slot_dim** (int，可选) -  一个 slot 的 embedding 维度，slot 用来表征一类特征的集合，在 pslib 模式下，通常我们通过 slot 区分特征 id，并从参数服务器 (pslib) 中提取它们的 embedding。embedding 的第一维是历史上这个 embedding 展示的次数。如果本 op 的输入是由这样的 embedding 连接而来，那么当这个特征 id 是新的或空的，则正则化结果可能不实际。为了避免这种情况，我们添加了 slot_dim 来定位并判断这一维是否为零。如果是的话，我们选择跳过正则化。默认值为 -1。
  - **sync_stats** (bool，可选) - 在多 GPU 卡的场景下可以使用，用来同步多卡间的 summary 信息。默认值为 False。
  - **summary_decay_rate** (float，可选) - 更新 summary 信息时的衰减率。默认值为 0.9999999。
  - **enable_scale_and_shift** (bool，可选) - 在分布式全局正则化后是否做像 batchnorm 一样做 scale&shift 的操作。默认值为 False。

返回
::::::::::::
Tensor，是对输入数据进行正则化后的结果。


代码示例
::::::::::::

COPY-FROM: paddle.static.nn.data_norm
