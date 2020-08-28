.. _cn_api_fluid_dygraph_InstanceNorm:

InstanceNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.InstanceNorm(num_channels, epsilon=1e-05, param_attr=None, bias_attr=None, dtype='float32') 

该接口用于构建 ``InstanceNorm`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。

可用作卷积和全连接操作的实例正则化函数，根据每个样本的每个通道的均值和方差信息进行正则化。该层需要的数据格式如下：

NCHW[batch,in_channels,in_height,in_width]

更多详情请参考 : `Instance Normalization: The Missing Ingredient for Fast Stylization <https://arxiv.org/pdf/1607.08022.pdf>`_

``input`` 是mini-batch的输入。

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mean of each channel in each sample in a batch  \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// variance of each channel in each sample a batch  \\
    \hat{x_i}          &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta                                                      \quad &// scale-and-shift


参数：
    - **num_channels** （int）- 指明输入 ``Tensor`` 的通道数量。
    - **epsilon** （float，默认1e-05）- 为了当前输入做标准化时得到稳定的结果而加在的分母上的扰动值。默认值为1e-5。
    - **param_attr** （ParamAttr|None） - instance_norm 权重参数的属性，可以设置为None或者一个ParamAttr的类（ParamAttr中可以指定参数的各种属性）。 如果设为None，则默认的参数初始化为1.0。如果在ParamAttr指定了属性时, instance_norm创建相应属性的param_attr（权重）参数。默认：None。
    - **bias_attr** （ParamAttr|None） - instance_norm 偏置参数的属性，可以设置为None或者一个ParamAttr的类（ParamAttr中可以指定参数的各种属性）。如果设为None，默认的参数初始化为0.0。如果在ParamAttr指定了参数的属性时, instance_norm创建相应属性的bias_attr（偏置）参数。默认：None。
    - **dtype** （string，默认float32）- 指明输入 ``Tensor`` 的数据类型，可以为float32或float64。默认：float32。

返回：无

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph.base import to_variable
    import numpy as np
    import paddle

    # x's shape is [1, 3, 1, 2] 
    x = np.array([[[[1.0, 8.0]], [[10.0, 5.0]], [[4.0, 6.0]]]]).astype('float32')
    with fluid.dygraph.guard():
        x = to_variable(x)
        instanceNorm = paddle.nn.InstanceNorm(3)
        ret = instanceNorm(x)
        # ret's shape is [1, 3, 1, 2]; value is [-1 1 0.999999 -0.999999 -0.999995 0.999995]
        print(ret)

