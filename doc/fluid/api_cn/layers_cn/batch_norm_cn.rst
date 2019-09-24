.. _cn_api_fluid_layers_batch_norm:

batch_norm
-------------------------------

.. py:function:: paddle.fluid.layers.batch_norm(input, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, data_layout='NCHW', in_place=False, name=None, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, fuse_with_relu=False, use_global_stats=False)

批正则化层（Batch Normalization Layer）

可用作卷积和全连接操作的批正则化函数，根据当前批次数据按通道计算的均值和方差进行正则化。该层需要的数据格式如下：

1.NHWC[batch,in_height,in_width,in_channels]
2.NCHW[batch,in_channels,in_height,in_width]

更多详情请参考 : `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_

``input`` 是mini-batch的输入。

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mini-batch-mean \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// mini-batch-variance \\
    \hat{x_i}          &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta                                                      \quad &// scale-and-shift

    moving\_mean = moving\_mean * momentum + mini\_batch\_mean * (1. - momentum)                     \global mean
    moving\_variance = moving\_variance * momentum + mini\_batch\_var * (1. - momentum)              \global variance

moving_mean和moving_var是训练过程中统计得到的全局均值和方差，在预测或者评估中使用。

当use_global_stats = True时， :math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 不是一个minibatch的统计数据。 它们是全局（或运行）统计数据（moving_mean和moving_variance），通常来自预先训练好的模型。训练和测试（或预测）具有相同的行为：

.. math::

    \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
    \sigma_{\beta}^{2} + \epsilon}}  \\
    y_i &\gets \gamma \hat{x_i} + \beta



参数：
    - **input** (Variable) - batch_norm算子的输入特征，是一个Variable类型，输入维度可以是 2, 3, 4, 5。
    - **act** （string）- 激活函数类型，可以是leaky_realu、relu、prelu等。默认：None。
    - **is_test** （bool） - 指示它是否在测试阶段，非训练阶段使用训练过程中统计到的全局均值和全局方差。默认：False。
    - **momentum** （float）- 此值用于计算 moving_mean 和 moving_var。更新公式为:  :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)` ， :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)` ， 默认：0.9。
    - **epsilon** （float）- 加在分母上为了数值稳定的值。默认：1e-5。
    - **param_attr** （ParamAttr|None） - batch_norm 权重参数的属性，可以设置为None或者一个ParamAttr的类（ParamAttr中可以指定参数的各种属性）。 如果设为None，则默认的参数初始化为1.0。如果在ParamAttr指定了属性时, batch_norm创建相应属性的param_attr（权重）参数。默认：None。
    - **bias_attr** （ParamAttr|None） - batch_norm bias参数的属性，可以设置为None或者一个ParamAttr的类（ParamAttr中可以指定参数的各种属性）。如果设为None，默认的参数初始化为0.0。如果在ParamAttr指定了参数的属性时, batch_norm创建相应属性的bias_attr（偏置）参数。默认：None。
    - **data_layout** （string) - 指定输入数据格式，数据格式可以为NCHW或者NHWC。默认：NCHW。
    - **in_place** （bool）- batch_norm的输出复用输入的tensor，可以节省显存。默认：False。
    - **name** （string，默认None）- 该层名称（可选）。若设为None，则自动为该层命名。默认：None。
    - **moving_mean_name** （string）- moving_mean的名称，存储全局均值。如果将其设置为None, ``batch_norm`` 将随机命名全局均值；否则， ``batch_norm`` 将命名全局均值为 ``moving_mean_name`` 。默认：None。
    - **moving_variance_name** （string）- moving_variance的名称，存储全局变量。如果将其设置为None, ``batch_norm`` 将随机命名全局方差；否则， ``batch_norm`` 将命名全局方差为 ``moving_variance_name`` 。默认：None。
    - **do_model_average_for_mean_and_var** （bool，默认False）- 是否为mean和variance做模型均值。
    - **fuse_with_relu** （bool）- 如果为True，batch_norm后该操作符执行relu。默认：None。
    - **use_global_stats** （bool） – 是否使用全局均值和方差。 在预测或测试模式下，将use_global_stats设置为true或将is_test设置为true，并且行为是等效的。 在训练模式中，当设置use_global_stats为True时，在训练期间也使用全局均值和方差。默认：False。

返回： 维度和输入相同的Tensor，在输入中运用批正则后的结果。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    x = fluid.layers.data(name='x', shape=[3, 7, 3, 7], dtype='float32', append_batch_size=False)
    hidden1 = fluid.layers.fc(input=x, size=200)
    param_attr = fluid.ParamAttr(name='batch_norm_w', initializer=fluid.initializer.Constant(value=1.0))
    bias_attr = fluid.ParamAttr(name='batch_norm_b', initializer=fluid.initializer.Constant(value=0.0))
    hidden2 = fluid.layers.batch_norm(input=hidden1, param_attr = param_attr, bias_attr = bias_attr)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    np_x = np.random.random(size=(3, 7, 3, 7)).astype('float32')
    output = exe.run(feed={"x": np_x}, fetch_list = [hidden2])
    print(output)


