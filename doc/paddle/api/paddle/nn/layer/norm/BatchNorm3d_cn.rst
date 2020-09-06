.. _cn_api_nn_BatchNorm3d:

BatchNorm3d
-------------------------------

.. py:class:: paddle.nn.BatchNorm3d(num_features, momentum=0.9, epsilon=1e-05, weight_attr=None, bias_attr=None, data_format='NCDHW', track_running_stats=True, name=None):


该接口用于构建 ``BatchNorm3d`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。可以处理4D的Tensor, 实现了批归一化层（Batch Normalization Layer）的功能，可用作卷积和全连接操作的批归一化函数，根据当前批次数据按通道计算的均值和方差进行归一化。更多详情请参考 : `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_

当训练时 :math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 是minibatch的统计数据。计算公式如下：

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mini-batch-mean \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// mini-batch-variance \\

- :math:`x` : 批输入数据
- :math:`m` : 当前批次数据的大小

当预测时，track_running_stats = True :math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 是全局（或运行）统计数据（moving_mean和moving_variance），通常来自预先训练好的模型。计算公式如下：

.. math::

    moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global mean \\
    moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global variance \\

归一化函数公式如下：

.. math::

    \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}} \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta \quad &// scale-and-shift \\

- :math:`\epsilon` : 添加较小的值到方差中以防止除零
- :math:`\gamma` : 可训练的比例参数
- :math:`\beta` : 可训练的偏差参数

参数：
    - **num_features** (int) - 指明输入 ``Tensor`` 的通道数量。
    - **epsilon** (float, 可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **momentum** (float, 可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var`` 。默认值：0.9。更新公式如上所示。
    - **weight_attr** (ParamAttr|bool, 可选) - 指定权重参数属性的对象。如果为False, 则表示每个通道的伸缩固定为1，不可改变。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_ParamAttr` 。
    - **bias_attr** (ParamAttr, 可选) - 指定偏置参数属性的对象。如果为False, 则表示每一个通道的偏移固定为0，不可改变。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_ParamAttr` 。
    - **data_format** (string, 可选) - 指定输入数据格式，数据格式可以为“NCDHW"。默认值：“NCDHW”。
    - **track_running_stats** (bool, 可选) – 指示是否使用全局均值和方差。在训练时，设置为True表示在训练期间将保存全局均值和方差用于推理。推理时此属性只能设置为True。默认值：True。
    - **name** (string, 可选) – BatchNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。


返回：无

形状：
    - input: 形状为（批大小，通道数, 维度，高度，宽度）的5-D Tensor。
    - output: 和输入形状一样。

.. note::
目前训练时设置track_running_stats为False是无效的，实际还是会按照True的方案保存全局均值和方差。之后的版本会修复此问题。
    

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    np.random.seed(123)
    x_data = np.random.random(size=(2, 1, 2, 2, 3)).astype('float32')
    x = paddle.to_tensor(x_data) 
    batch_norm = paddle.nn.BatchNorm3d(1)
    batch_norm_out = batch_norm(x)

    print(batch_norm_out.numpy())

