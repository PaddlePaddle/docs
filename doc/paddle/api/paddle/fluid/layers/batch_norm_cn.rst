.. _cn_api_fluid_layers_batch_norm:

batch_norm
-------------------------------


.. py:function:: paddle.fluid.layers.batch_norm(input, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, data_layout='NCHW', in_place=False, name=None, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, use_global_stats=False)

:api_attr: 声明式编程模式（静态图)



批正则化层（Batch Normalization Layer）

可用作卷积和全连接操作的批正则化函数，根据当前批次数据按通道计算的均值和方差进行正则化。该层需要的数据格式如下：

1.NHWC[batch,in_height,in_width,in_channels]
2.NCHW[batch,in_channels,in_height,in_width]

更多详情请参考 : `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_

``input`` 是mini-batch的输入。

.. math::
    \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i  \qquad &//\
    \ mini-batch\ mean \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2  \qquad &//\
    \ mini-batch\ variance \\
    \hat{x_i}  &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \qquad &//\ normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta  \qquad &//\ scale\ and\ shift

    moving\_mean = moving\_mean * momentum + mini\_batch\_mean * (1. - momentum) \\          
    moving\_variance = moving\_variance * momentum + mini\_batch\_var * (1. - momentum)     

moving_mean和moving_var是训练过程中统计得到的全局均值和方差，在预测或者评估中使用。
`is_test` 参数只能用于测试或者评估阶段，如果想在训练阶段使用预训练模型的全局均值和方差的话，可以设置 `use_global_stats=True`.

当use_global_stats = True时， :math:`\mu_{\beta}` 和 :math:`\sigma_{\beta}^{2}` 不是一个minibatch的统计数据。 它们是全局（或运行）统计数据（moving_mean和moving_variance），通常来自预先训练好的模型。训练和测试（或预测）具有相同的行为：

.. math::

    \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
    \sigma_{\beta}^{2} + \epsilon}}  \\
    y_i &\gets \gamma \hat{x_i} + \beta



参数：
    - **input** (Variable) - batch_norm算子的输入特征，是一个Variable类型，输入维度可以是 2, 3, 4, 5。数据类型：flaot16, float32, float64。
    - **act** （string）- 激活函数类型，可以是leaky_realu、relu、prelu等。默认：None。
    - **is_test** （bool） - 指示它是否在测试阶段，非训练阶段使用训练过程中统计到的全局均值和全局方差。默认：False。
    - **momentum** （float|Variable）- 此值用于计算 moving_mean 和 moving_var，是一个float类型或者一个shape为[1]，数据类型为float32的Variable类型。更新公式为:  :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)` ， :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)` ， 默认：0.9。
    - **epsilon** （float）- 加在分母上为了数值稳定的值。默认：1e-5。
    - **param_attr** (ParamAttr|None) ：指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。batch_norm算子默认的权重初始化是1.0。
    - **bias_attr** （ParamAttr|None）- 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。batch_norm算子默认的偏置初始化是0.0。
    - **data_layout** （string) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。
    - **in_place** （bool）- batch_norm的输出复用输入的tensor，可以节省显存。默认：False。
    - **name** (str|None) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值为None。
    - **moving_mean_name** （string）- moving_mean的名称，存储全局均值。如果将其设置为None, ``batch_norm`` 将随机命名全局均值；否则， ``batch_norm`` 将命名全局均值为 ``moving_mean_name`` 。默认：None。
    - **moving_variance_name** （string）- moving_variance的名称，存储全局变量。如果将其设置为None, ``batch_norm`` 将随机命名全局方差；否则， ``batch_norm`` 将命名全局方差为 ``moving_variance_name`` 。默认：None。
    - **do_model_average_for_mean_and_var** （bool，默认False）- 是否为mean和variance做模型均值。
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

.. code-block:: python

    # batch_norm with momentum as Variable
    import paddle.fluid as fluid
    import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler
    
    def get_decay_momentum(momentum_init, decay_steps, decay_rate):
        global_step = lr_scheduler._decay_step_counter()
        momentum = fluid.layers.create_global_var(
            shape=[1],
            value=float(momentum_init),
            dtype='float32',
            # set persistable for save checkpoints and resume
            persistable=True,
            name="momentum")
        div_res = global_step / decay_steps
        decayed_momentum = momentum_init * (decay_rate**div_res)
        fluid.layers.assign(decayed_momentum, momentum)
        
        return momentum
    
    x = fluid.data(name='x', shape=[3, 7, 3, 7], dtype='float32')
    hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
    momentum = get_decay_momentum(0.9, 1e5, 0.9)
    hidden2 = fluid.layers.batch_norm(input=hidden1, momentum=momentum)

