.. _cn_api_fluid_dygraph_BatchNorm:

BatchNorm
-------------------------------

.. py:class:: paddle.nn.BatchNorm(num_channels, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, dtype='float32', data_layout='NCHW', in_place=False, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, use_global_stats=False, trainable_statistics=False)




构建 ``BatchNorm`` 类的一个可调用对象，具体用法参照 ``代码示例``。其中实现了批归一化层（Batch Normalization Layer）的功能，可用作卷积和全连接操作的批归一化函数，根据当前批次数据按通道计算的均值和方差进行归一化。更多详情请参考：`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_

当 use_global_stats = False 时，:math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 是 minibatch 的统计数据。计算公式如下：

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mini-batch-mean \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// mini-batch-variance \\

- :math:`x`：批输入数据
- :math:`m`：当前批次数据的大小

当 use_global_stats = True 时，:math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 是全局（或运行）统计数据（moving_mean 和 moving_variance），通常来自预先训练好的模型。计算公式如下：

.. math::

    moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global mean \\
    moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global variance \\

归一化函数公式如下：

.. math::

    \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}} \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta \quad &// scale-and-shift \\

- :math:`\epsilon`：添加较小的值到方差中以防止除零
- :math:`\gamma`：可训练的比例参数
- :math:`\beta`：可训练的偏差参数

参数
::::::::::::

    - **num_channels** (int) - 指明输入 ``Tensor`` 的通道数量。
    - **act** (str，可选) - 应用于输出上的激活函数，如 tanh、softmax、sigmoid，relu 等，支持列表请参考 :ref:`api_guide_activations`，默认值为 None。
    - **is_test** (bool，可选) - 指示是否在测试阶段，非训练阶段使用训练过程中统计到的全局均值和全局方差。默认值：False。
    - **momentum** (float，可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var``。默认值：0.9。更新公式如上所示。
    - **epsilon** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **dtype** (str，可选) - 指明输入 ``Tensor`` 的数据类型，可以为 float32 或 float64。默认值：float32。
    - **data_layout** (str，可选) - 指定输入数据格式，数据格式可以为 ``"NCHW"`` 或 ``"NHWC"``，其中 N 是批大小，C 是通道数，H 是特征高度，W 是特征宽度。默认值为 ``"NCHW"``。
    - **in_place** (bool，可选) - 指示 ``batch_norm`` 的输出是否可以复用输入内存。默认值：False。
    - **moving_mean_name** (str，可选) - ``moving_mean`` 的名称，存储全局均值。如果将其设置为 None, ``batch_norm`` 将随机命名全局均值；否则，``batch_norm`` 将命名全局均值为 ``moving_mean_name``。默认值：None。
    - **moving_variance_name** (string，可选) - ``moving_var`` 的名称，存储全局方差。如果将其设置为 None, ``batch_norm`` 将随机命名全局方差；否则，``batch_norm`` 将命名全局方差为 ``moving_variance_name``。默认值：None。
    - **do_model_average_for_mean_and_var** (bool，可选) - 指示是否为 mean 和 variance 做模型均值。默认值：False。
    - **use_global_stats** (bool，可选) – 指示是否使用全局均值和方差。在预测或测试模式下，将 ``use_global_stats`` 设置为 true 或将 ``is_test`` 设置为 true，这两种行为是等效的。在训练模式中，当设置 ``use_global_stats`` 为 True 时，在训练期间也将使用全局均值和方差。默认值：False。
    - **trainable_statistics** (bool，可选) - eval 模式下是否计算 mean 均值和 var 方差。eval 模式下，trainable_statistics 为 True 时，由该批数据计算均值和方差。默认值：False。

返回
::::::::::::
无

代码示例
::::::::::::

COPY-FROM: paddle.nn.BatchNorm
