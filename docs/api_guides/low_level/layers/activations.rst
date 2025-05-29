.. _api_guide_activations:

###################
激活函数
###################

激活函数将非线性的特性引入到神经网络当中。

PaddlePaddle 对大部分的激活函数进行了支持，其中有:

* :ref:`cn_api_paddle_nn_functional_celu`
* :ref:`cn_api_paddle_nn_functional_elu`
* :ref:`cn_api_paddle_exp`
* :ref:`cn_api_paddle_nn_functional_glu`
* :ref:`cn_api_paddle_nn_functional_gumbel_softmax`
* :ref:`cn_api_paddle_nn_functional_hardshrink`
* :ref:`cn_api_paddle_nn_functional_hardsigmoid`
* :ref:`cn_api_paddle_nn_functional_hardswish`
* :ref:`cn_api_paddle_nn_functional_hardtanh`
* :ref:`cn_api_paddle_nn_functional_leaky_relu`
* :ref:`cn_api_paddle_nn_functional_log_sigmoid`
* :ref:`cn_api_paddle_nn_functional_log_softmax`
* :ref:`cn_api_paddle_nn_functional_maxout`
* :ref:`cn_api_paddle_nn_functional_mish`
* :ref:`cn_api_paddle_pow`
* :ref:`cn_api_paddle_nn_functional_prelu`
* :ref:`cn_api_paddle_nn_functional_relu`
* :ref:`cn_api_paddle_nn_functional_relu6`
* :ref:`cn_api_paddle_nn_functional_rrelu`
* :ref:`cn_api_paddle_nn_functional_selu`
* :ref:`cn_api_paddle_nn_functional_sigmoid`
* :ref:`cn_api_paddle_nn_functional_silu`
* :ref:`cn_api_paddle_nn_functional_softmax`
* :ref:`cn_api_paddle_nn_functional_softplus`
* :ref:`cn_api_paddle_nn_functional_softshrink`
* :ref:`cn_api_paddle_nn_functional_softsign`
* :ref:`cn_api_paddle_stanh`
* :ref:`cn_api_paddle_nn_functional_swish`
* :ref:`cn_api_paddle_tanh`
* :ref:`cn_api_paddle_nn_functional_thresholded_relu`
* :ref:`cn_api_paddle_nn_functional_tanhshrink`


**PaddlePaddle 应用激活函数的方式如下：**

PaddlePaddle 为每个 Activation 提供了接口，可以显式调用。以下是一个示例，展示如何在卷积操作之后应用 ReLU 激活函数：

.. code-block:: python

    conv2d = paddle.nn.functional.conv2d(x, weight, stride=1, padding=1) # 卷积
    relu1 = paddle.nn.functional.relu(conv2d) # 使用 ReLu 激活函数
