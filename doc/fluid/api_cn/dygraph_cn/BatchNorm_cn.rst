.. _cn_api_fluid_dygraph_BatchNorm:

BatchNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.BatchNorm(name_scope, num_channels, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, dtype='float32', data_layout='NCHW', in_place=False, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, fuse_with_relu=False, use_global_stats=False, trainable_statistics=False)

该OP实现了批正则化层（Batch Normalization Layer），可用做conv2d和fc层的正则化函数。更多详情请参考 : `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_

其中 ``input`` 是mini-batch的输入特征。计算过程如下：

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mini-batch-mean \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// mini-batch-variance \\
    \hat{x_i}          &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta                                                      \quad &// scale-and-shift

当use_global_stats = True时，:math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 不是一个minibatch的统计数据。 它们是全局（或运行）统计数据。 （它通常来自预先训练好的模型。）训练和测试（或预测）具有相同的行为：

.. math::

    \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
    \sigma_{\beta}^{2} + \epsilon}}  \\
    y_i &\gets \gamma \hat{x_i} + \beta


参数：
    - **name_scope** (str) - 该类的名称
    - **act** （str, 默认None）- 激活函数类型，linear、relu、prelu等。
    - **is_test** （bool, 默认False） - 指示是否在测试阶段。
    - **momentum** （float, 默认0.9）- 此值用于计算 ``moving_mean`` 和 ``moving_var`` 。更新公式为:  :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)` ， :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)` 。
    - **epsilon** (float, 默认1e-05) - 为了数值稳定加在分母上的值。
    - **param_attr** (ParamAttr, 默认None) - ``batch_norm`` 中 ``scale`` 参数的属性， ``batch_norm`` 将利用 ``param_attr`` 属性来创建ParamAttr实例。如果没有设置 ``param_attr`` 的初始化函数，参数初始化为1.0。
    - **bias_attr** (ParamAttr，默认None) - ``batch_norm`` 中 ``bias`` 参数的属性， ``batch_norm`` 将利用 ``bias_attr`` 属性来创建ParamAttr实例。如果没有设置 ``bias_attr`` 的初始化函数，参数初始化为0.0。
    - **data_layout** (string, 默认"NCHW") - 值可以是"NCHW"或者"NHWC"，用来指示 ``input`` 参数输入数据的布局。
    - **in_place** (bool, 默认False) - 指示 ``batch_norm`` 的输出是否可以复用输入内存。
    - **moving_mean_name** (string, 默认None) - ``moving_mean`` 的名称，存储全局平均值。如果将其设置为None, ``batch_norm`` 将随机命名全局平均值；否则， ``batch_norm`` 将命名全局平均值为 ``moving_mean_name`` 。
    - **moving_variance_name** (string, 默认None) - ``moving_var`` 的名称，存储全局方差。如果将其设置为None, ``batch_norm`` 将随机命名全局方差；否则， ``batch_norm`` 将命名全局方差为 ``moving_variance_name`` 。
    - **do_model_average_for_mean_and_var** (bool, 默认False) - 指示是否为mean和variance做模型均值。
    - **fuse_with_relu** (bool) - 如果为True，将在执行 ``batch_norm`` 后执行 ``relu`` 操作。
    - **use_global_stats** (bool, Default False) – 指示是否使用全局均值和方差。在预测或测试模式下，将 ``use_global_stats`` 设置为true或将 ``is_test`` 设置为true，这两种行为是等效的。在训练模式中，当设置 ``use_global_stats`` 为True时，在训练期间也将使用全局均值和方差。
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



