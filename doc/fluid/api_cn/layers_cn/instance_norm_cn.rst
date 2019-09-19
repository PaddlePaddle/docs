.. _cn_api_fluid_layers_instance_norm:

instance_norm
-------------------------------

.. py:function:: paddle.fluid.layers.instance_norm(input, epsilon=1e-05, param_attr=None, bias_attr=None, name=None)


可用作卷积和全链接操作的正则化函数。该层需要的数据格式如下：

NCHW[batch,in_channels,in_height,in_width]

更多详情请参考 : `Instance Normalization: The Missing Ingredient for Fast Stylization <https://arxiv.org/pdf/1607.08022.pdf>`_

``input`` 是mini-batch的输入。

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mean of each channel in each sample in a batch  \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// variance of each channel in each sample a batch  \\
    \hat{x_i}          &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta                                                      \quad &// scale-and-shift


参数：
    - **input** (Variable) - 输入，输入的维度可以为 2, 3, 4, 5。
    - **epsilon** （float，默认1e-05）-为了当前输入做标准化时得到稳定的结果而加在的分母上的扰动值。默认值为1e-5。
    - **param_attr** （ParamAttr|None） - instance_norm gamma参数的属性，可以设置为None或者一个ParamAttr。 如果设为None，instance_norm自动创建ParamAttr为param_attr，param_attr默认的参数初始化为Xavier。如果在ParamAttr指定了param_attr的属性时, instance_norm创建相应属性的param_attr。默认：None。
    - **bias_attr** （ParamAttr|None） - instance_norm beta参数的属性，可以设置为None或者一个ParamAttr。如果设为None，instance_norm自动创建ParamAttr为bias_attr，bias_attr默认的参数初始化为0。如果在ParamAttr指定了bias_attr的参数时, instance_norm创建相应属性的bias_attr。默认：None。
    - **name** （string，默认None）- 该层名称（可选）。若设为None，则自动为该层命名。

返回： 张量，在输入中运用instance normalization后的结果

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python
    
    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[3, 7, 3, 7], dtype='float32', append_batch_size=False)
    hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
    hidden2 = fluid.layers.instance_norm(input=hidden1)











