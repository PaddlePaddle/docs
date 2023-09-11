.. _cn_api_paddle_sparse_nn_SyncBatchNorm:

SyncBatchNorm
-------------------------------

.. py:class:: paddle.sparse.nn.SyncBatchNorm(num_features, epsilon=1e-5, momentum=0.9, weight_attr=None, bias_attr=None, data_format='NCHW', name=None)

构建 ``SyncBatchNorm`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。实现了跨卡 GPU 同步的批归一化(Cross-GPU Synchronized Batch Normalization Layer)的功能，可用在其他层（类似卷积层和全连接层）之后进行归一化操作。根据所有 GPU 同一批次的数据按照通道计算的均值和方差进行归一化。更多详情请参考：`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_ 。

当模型处于训练模式时，:math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 是所有 GPU 上同一 minibatch 的统计数据。计算公式如下：

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mini-batch-mean \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// mini-batch-variance \\

- :math:`x` ：所有 GPU 上同一批输入数据
- :math:`m` ：所有 GPU 上同一批次数据的大小

当模型处于评估模式时，:math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 是全局（或运行）统计数据（moving_mean 和 moving_variance，这两个统计量通常来自预先训练好的模型）。计算公式如下：

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

.. note::

    如果您想用容器封装您的模型，而且您的模型在预测阶段中包含 ``SyncBatchNorm`` 这个算子的话，请使用 ``nn.LayerList`` 或者 ``nn.Sequential`` 而不要直接使用 ``list`` 来封装模型。

参数
::::::::::::

    - **num_features** (int) - 指明输入 ``Tensor`` 的通道数量。
    - **epsilon** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **momentum** (float，可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var`` 。默认值：0.9。更新公式如上所示。
    - **weight_attr** (ParamAttr|bool，可选) - 指定权重参数属性的对象。如果设置为 ``False`` ，则表示本层没有可训练的权重参数。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **bias_attr** (ParamAttr|bool，可选) - 指定偏置参数属性的对象。如果设置为 ``False`` ，则表示本层没有可训练的偏置参数。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **data_format** (string，可选) - 指定输入数据格式，数据格式可以为“NCHW"。默认值：“NCHW”。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

形状
::::::::::::

    - input：一个二维到五维的 ``Tensor`` 。
    - output：和 input 相同形状的 ``Tensor`` 。

代码示例
::::::::::::

COPY-FROM: paddle.sparse.nn.SyncBatchNorm

方法
:::::::::
convert_sync_batchnorm(layer)
'''''''''''''''''''''''''''''

把 ``BatchNorm`` 层转换为 ``SyncBatchNorm`` 层。

参数
::::::::::::

    - **layer** (paddle.nn.Layer) - 包含一个或多个 ``BatchNorm`` 层的模型。

返回
::::::::::::

    如果原始模型中有 ``BatchNorm`` 层，则把 ``BatchNorm`` 层转换为 ``SyncBatchNorm`` 层的原始模型。

代码示例
::::::::::::

COPY-FROM: paddle.sparse.nn.SyncBatchNorm.convert_sync_batchnorm
