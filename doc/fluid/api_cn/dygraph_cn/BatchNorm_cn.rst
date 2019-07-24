.. _cn_api_fluid_dygraph_BatchNorm:

BatchNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.BatchNorm(name_scope, num_channels, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, dtype='float32', data_layout='NCHW', in_place=False, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, fuse_with_relu=False, use_global_stats=False, trainable_statistics=False)


批正则化层（Batch Normalization Layer）

可用作conv2d和全连接操作的正则化函数。该层需要的数据格式如下：

1.NHWC[batch,in_height,in_width,in_channels]
2.NCHW[batch,in_channels,in_height,in_width]

更多详情请参考 : `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_

``input`` 是mini-batch的输入特征。

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mini-batch-mean \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// mini-batch-variance \\
    \hat{x_i}          &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta                                                      \quad &// scale-and-shift

当use_global_stats = True时， :math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 不是一个minibatch的统计数据。 它们是全局（或运行）统计数据。 （它通常来自预训练模型）。训练和测试（或预测）具有相同的行为：

.. math::

    \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
    \sigma_{\beta}^{2} + \epsilon}}  \\
    y_i &\gets \gamma \hat{x_i} + \beta



参数：
    - **name_scope** (str) - 该类的名称
    - **act** （string，默认None）- 激活函数类型，linear|relu|prelu|...
    - **is_test** （bool,默认False） - 指示它是否在测试阶段。
    - **momentum** （float，默认0.9）- 此值用于计算 moving_mean and moving_var. 更新公式为:  :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum` :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum` ， 默认值0.9.
    - **epsilon** （float，默认1e-05）- 加在分母上为了数值稳定的值。默认值为1e-5。
    - **param_attr** （ParamAttr|None） - batch_norm参数范围的属性，如果设为None或者是ParamAttr的一个属性，batch_norm创建ParamAttr为param_attr。如果没有设置param_attr的初始化函数，参数初始化为Xavier。默认：None
    - **bias_attr** （ParamAttr|None） - batch_norm bias参数的属性，如果设为None或者是ParamAttr的一个属性，batch_norm创建ParamAttr为bias_attr。如果没有设置bias_attr的初始化函数，参数初始化为0。默认：None
    - **data_layout** （string,默认NCHW) - NCHW|NHWC。默认NCHW
    - **in_place** （bool，默认False）- 得出batch norm可复用记忆的输入和输出
    - **moving_mean_name** （string，默认None）- moving_mean的名称，存储全局Mean均值。 
    - **moving_variance_name** （string，默认None）- moving_variance的名称，存储全局方差。 
    - **do_model_average_for_mean_and_var** （bool，默认False）- 是否为mean和variance做模型均值
    - **fuse_with_relu** （bool）- 如果为True，batch norm后该操作符执行relu。默认为False。
    - **use_global_stats** （bool, Default False） – 是否使用全局均值和方差。 在预测或测试模式下，将use_global_stats设置为true或将is_test设置为true，并且行为是等效的。 在训练模式中，当设置use_global_stats为True时，在训练期间也使用全局均值和方差。
    - **trainable_statistics** （bool）- eval模式下是否计算mean均值和var方差。eval模式下，trainable_statistics为True时，由该批数据计算均值和方差。默认为False。

返回： 张量，在输入中运用批正则后的结果

返回类型：变量（Variable）

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    with fluid.dygraph.guard():
        fc = fluid.FC('fc', size=200, param_attr='fc1.w')
        hidden1 = fc(x)
        batch_norm = fluid.BatchNorm("batch_norm", 10)
        hidden2 = batch_norm(hidden1)



