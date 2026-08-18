.. _cn_api_paddle_sparse_nn_BatchNorm:

BatchNorm
-------------------------------

.. py:class:: paddle.sparse.nn.BatchNorm(num_features, momentum=0.9, epsilon=1e-05, weight_attr=None, bias_attr=None, data_format="NDHWC", use_global_stats=None, name=None)


构建稀疏 ``BatchNorm`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。可处理 NHWC 格式的 4-D 或 NDHWC 格式的 5-D SparseCooTensor，实现了批归一化层（Batch Normalization Layer）的功能，可用作卷积和全连接操作的批归一化函数，根据当前批次数据按通道计算的均值和方差进行归一化。更多详情请参考： `Batch Normalization : Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_ 。

当 use_global_stats = False 时 :math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 是 minibatch 的统计数据。计算公式如下：

.. math::

    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mini-batch-mean \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// mini-batch-variance \\

- :math:`x` ：批输入数据
- :math:`m` ：当前批次数据的大小

当 use_global_stats = True :math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 是全局（或运行）统计数据（moving_mean 和 moving_variance），通常来自预先训练好的模型。计算公式如下：

.. math::

    moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global mean \\
    moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global variance \\

归一化函数公式如下：

.. math::

    \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}} \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta \quad &// scale-and-shift \\

- :math:`\epsilon` ：添加较小的值到方差中以防止除零
- :math:`\gamma` ：可训练的比例参数
- :math:`\beta` ：可训练的偏差参数

参数
::::::::::::

    - **num_features** (int) - 指明输入 ``Tensor`` 的通道数量。
    - **momentum** (float，可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var`` 。默认值：0.9。更新公式如上所示。
    - **epsilon** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **weight_attr** (ParamAttr|bool，可选) - 指定权重参数属性的对象。如果为 False，则表示每个通道的伸缩固定为 1，不可改变。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **bias_attr** (ParamAttr|bool，可选) - 指定偏置参数属性的对象。如果为 False，则表示每一个通道的偏移固定为 0，不可改变。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **data_format** (string，可选) - 指定输入数据格式，数据格式可以为 ``"NDHWC"`` 或 ``"NHWC"``。默认值为 ``"NDHWC"``。
    - **use_global_stats** (bool|None，可选) – 指示是否使用全局均值和方差。若为 False，使用一个 mini-batch 的统计数据；若为 True，使用全局统计数据；若为 None，则在预测阶段使用全局统计数据、在训练阶段使用一个 mini-batch 的统计数据。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。


返回
::::::::::::
无

形状
::::::::::::

    - input：NDHWC 格式的 5-D 或 NHWC 格式的 4-D SparseCooTensor。
    - output：与输入具有相同形状和布局的 SparseCooTensor。

代码示例
::::::::::::

COPY-FROM: paddle.sparse.nn.BatchNorm
