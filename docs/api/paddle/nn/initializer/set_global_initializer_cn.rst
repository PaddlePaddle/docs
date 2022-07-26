.. _cn_api_nn_initializer_set_global_initializer:

set_global_initializer
-------------------------------

.. py:function:: paddle.nn.initializer.set_global_initializer(weight_init, bias_init=None)

该API用于设置Paddle框架中全局的参数初始化方法。该API只对位于其后的代码生效。

模型参数为模型中的weight和bias统称，在fluid中对应fluid.Parameter类，继承自fluid.Variable，是一种可持久化的variable。
该API的设置仅对模型参数生效，对通过 :ref:`cn_api_fluid_layers_create_global_var` 、 :ref:`cn_api_fluid_layers_create_tensor` 等API创建的变量不会生效。

如果创建网络层时还通过 ``param_attr`` 、 ``bias_attr`` 设置了初始化方式，这里的全局设置将不会生效，因为其优先级更低。

参数
::::::::::::

    - **weight_init** (Initializer) - 设置框架的全局的weight参数初始化方法。
    - **bias_init** (Initializer，可选) - 设置框架的全局的bias参数初始化方法。默认：None。

返回
::::::::::::
无

代码示例
::::::::::::

COPY-FROM: paddle.nn.initializer.set_global_initializer