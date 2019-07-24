.. _cn_api_fluid_layers_data_norm:

data_norm
-------------------------------

.. py:function:: paddle.fluid.layers.data_norm(input, act=None, epsilon=1e-05, param_attr=None, data_layout='NCHW', in_place=False, name=None, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False)

**数据正则化层**

可用作conv2d和fully_connected操作的正则化函数。 此层所需的数据格式为以下之一：

1. NHWC [batch, in_height, in_width, in_channels]
2. NCHW [batch, in_channels, in_height, in_width]

:math:`input` 为一个mini-batch上的特征:

.. math::
        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &//\
        \ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \
        \mu_{\beta})^2 \qquad &//\ mini-batch\ variance \\
        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

参数:
  - **input** （variable） - 输入变量，它是一个LoDTensor。
  - **act** （string，默认None） - 激活函数类型，线性| relu | prelu | ...
  - **epsilon** （float，默认1e-05） -
  - **param_attr** （ParamAttr） - 参数比例的参数属性。
  - **data_layout** （string，默认NCHW） -  NCHW | NHWC
  - **in_place** （bool，默认值False） - 使data_norm的输入和输出复用同一块内存。
  - **name** （string，默认None） - 此层的名称（可选）。 如果设置为None，则将自动命名该层。
  - **moving_mean_name** （string，Default None） - 存储全局Mean的moving_mean的名称。
  - **moving_variance_name** （string，默认None） - 存储全局Variance的moving_variance的名称。
  - **do_model_average_for_mean_and_var** （bool，默认值为false） - 是否为mean和variance进行模型平均。

返回: 张量变量，是对输入数据进行正则化后的结果。

返回类型: Variable

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid

    hidden1 = fluid.layers.data(name="hidden1", shape=[200])
    hidden2 = fluid.layers.data_norm(name="hidden2", input=hidden1)







