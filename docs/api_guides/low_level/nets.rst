..  _api_guide_nets:

###########
复杂网络
###########

在处理复杂功能时，我们通常需要写大量的代码来构建复杂的 `神经网络 <https://zh.wikipedia.org/wiki/人工神经网络>`_ 。
因此，为了方便用户更加容易地搭建复杂网络模型，我们提供了一些比较常用的基本函数模块，以此来简化用户的代码量，从而降低开发成本。
这些模块通常是由细粒度的函数根据一定的逻辑拼接组合而成，实现代码请参考 `nets.py <https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/nets.py>`_ 。

1.simple_img_conv_pool
----------------------

:code:`simple_img_conv_pool` 是由 :ref:`cn_api_fluid_layers_conv2d` 与 :ref:`cn_api_fluid_layers_pool2d` 串联而成。
该模块在图像分类模型中广泛使用，比如应用在 `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_ 数字分类的问题。

API Reference 请参考 :ref:`cn_api_fluid_nets_simple_img_conv_pool`


2.img_conv_group
----------------

:code:`img_conv_group` 是由 :ref:`cn_api_fluid_layers_conv2d` , :ref:`cn_api_fluid_layers_batch_norm`, :ref:`cn_api_fluid_layers_dropout` 和 :ref:`cn_api_fluid_layers_pool2d` 组成。
该模块可以实现多个 :ref:`cn_api_fluid_layers_conv2d` , :ref:`cn_api_fluid_layers_batch_norm` 和 :ref:`cn_api_fluid_layers_dropout` 的串联单元与一个 :ref:`cn_api_fluid_layers_pool2d` 的组合。
其中， :ref:`cn_api_fluid_layers_conv2d` , :ref:`cn_api_fluid_layers_batch_norm` 和 :ref:`cn_api_fluid_layers_dropout` 的数量都可以分别控制，从而得到多样的组合。
该模块广泛使用在比较复杂的图像分类任务中，比如 `VGG <https://arxiv.org/pdf/1409.1556.pdf>`_ 。

API Reference 请参考 :ref:`cn_api_fluid_nets_img_conv_group`


3.sequence_conv_pool
--------------------

:code:`sequence_conv_pool` 是由 :ref:`cn_api_fluid_layers_sequence_conv` 与 :ref:`cn_api_fluid_layers_sequence_pool` 串联而成。
该模块在 `自然语言处理 <https://zh.wikipedia.org/wiki/自然语言处理>`_ 以及 `语音识别 <https://zh.wikipedia.org/wiki/语音识别>`_ 等领域均有广泛应用，
比如 `文本分类模型 <https://github.com/PaddlePaddle/models/blob/develop/PaddleNLP/text_classification/nets.py>`_ ,
`TagSpace <https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/tagspace/train.py>`_  以及
`Multi-view Simnet <https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/multiview_simnet/nets.py>`_ 等模型。

API Reference 请参考 :ref:`cn_api_fluid_nets_sequence_conv_pool`


4.glu
-----
:code:`glu` 全称 Gated Linear Units， 来源于论文 `Language Modeling with Gated Convolutional Networks <https://arxiv.org/pdf/1612.08083.pdf>`_ ，由 :ref:`cn_api_fluid_layers_split` ， :ref:`cn_api_fluid_layers_sigmoid` 和 :ref:`cn_api_fluid_layers_elementwise_mul` 组成。
它会把输入数据均分为 2 等份，并对第二部分求 `Sigmoid <https://en.wikipedia.org/wiki/Sigmoid_function>`_ , 然后再与第一部分数据求点乘得到输出。

API Reference 请参考 :ref:`cn_api_fluid_nets_glu`


5.scaled_dot_product_attention
------------------------------
:code:`scaled_dot_product_attention` 来源于论文 `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_ ，主要是由 :ref:`cn_api_fluid_layers_fc` 和 :ref:`cn_api_fluid_layers_softmax` 组成。
对于输入数据 :code:`Queries` , :code:`Key` 和 :code:`Values` 按照如下公式求出 :code:`Attention` 。

.. math::
 Attention(Q, K, V)= softmax(QK^\mathrm{T})V

该模块广泛使用在 `机器翻译 <https://zh.wikipedia.org/zh/机器翻译>`_ 的模型中，比如 `Transformer <https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/neural_machine_translation/transformer>`_ 。

API Reference 请参考 :ref:`cn_api_fluid_nets_scaled_dot_product_attention`
