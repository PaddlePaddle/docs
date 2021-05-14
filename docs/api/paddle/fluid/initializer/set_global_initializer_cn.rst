.. _cn_api_nn_initializer_set_global_initializer:

set_global_initializer
-------------------------------

.. py:function:: paddle.nn.initializer.set_global_initializer(weight_init, bias_init=None)

该API用于设置Paddle框架中全局的参数初始化方法。该API只对位于其后的代码生效。

模型参数为模型中的weight和bias统称，在fluid中对应fluid.Parameter类，继承自fluid.Variable，是一种可持久化的variable。
该API的设置仅对模型参数生效，对通过 :ref:`cn_api_fluid_layers_create_global_var` 、 :ref:`cn_api_fluid_layers_create_tensor` 等API创建的变量不会生效。

如果创建网络层时还通过 ``param_attr`` 、 ``bias_attr`` 设置了初始化方式，这里的全局设置将不会生效，因为其优先级更低。

参数：
    - **weight_init** (Initializer) - 设置框架的全局的weight参数初始化方法。
    - **bias_init** (Initializer，可选) - 设置框架的全局的bias参数初始化方法。默认：None。

返回：无

**代码示例**

.. code-block:: python

    import paddle
    import paddle.nn as nn

    nn.initializer.set_global_initializer(nn.initializer.Uniform(), nn.initializer.Constant())
    x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)

    # conv1的weight参数是通过Uniform来初始化
    # conv1的bias参数是通过Constant来初始化
    conv1 = nn.Conv2D(4, 6, (3, 3))
    y_var1 = conv1(x_var)

    # 如果同时设置了param_attr/bias_attr, 全局初始化将不会生效
    # conv2的weight参数是通过Xavier来初始化
    # conv2的bias参数是通过Normal来初始化
    conv2 = nn.Conv2D(4, 6, (3, 3), 
        weight_attr=nn.initializer.XavierUniform(),
        bias_attr=nn.initializer.Normal())
    y_var2 = conv2(x_var)
    
    # 取消全局参数初始化的设置
    nn.initializer.set_global_initializer(None)