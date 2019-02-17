.. _api_guide_nets_en:

################
Complex Network
################

When dealing with complex functions, we usually need to write a lot of code to build a complex `neural network <https://zh.wikipedia.org/wiki/artificial neural network>`_.
Therefore, in order to make it easier for users to build complex network models, we provide some common basic function modules to simplify the user's code and reduce development costs.
These modules are usually composed of fine-grained functions based on certain logical stitching. For implementation code, please refer to `nets.py <https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/nets .py>`_.

1.simple_img_conv_pool
----------------------

:code:`simple_img_conv_pool` is concatenated by :ref:`api_fluid_layers_conv2d` and :ref:`api_fluid_layers_pool2d`.
This module is widely used in image classification models, such as kapplied in the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_ number classification.

For API Reference, splease refer to :ref:`api_fluid_nets_simple_img_conv_pool`


2.img_conv_group
----------------

:code:`img_conv_group` is composed of :ref:`api_fluid_layers_conv2d` , :ref:`api_fluid_layers_batch_norm`, :ref:`api_fluid_layers_dropout` and :ref:`api_fluid_layers_pool2d`.
This module can implement multiple combinations of :ref:`api_fluid_layers_conv2d` , :ref:`api_fluid_layers_batch_norm` and :ref:`api_fluid_layers_dropout` and a :ref:`api_fluid_layers_pool2d`.
Among them, the number of :ref:`api_fluid_layers_conv2d` , :ref:`api_fluid_layers_batch_norm` and :ref:`api_fluid_layers_dropout` can be controlled separately, resulting in various combinations.
This module is widely used in more complex image classification tasks, such as `VGG <https://arxiv.org/pdf/1409.1556.pdf>`_.

For API Reference, please refer to :ref:`api_fluid_nets_img_conv_group`


3.sequence_conv_pool
--------------------

:code:`sequence_conv_pool` is concatenated by :ref:`api_fluid_layers_sequence_conv` and :ref:`api_fluid_layers_sequence_pool`.
The module is widely used in the field of `natural language processing <https://zh.wikipedia.org/wiki/natural language processing>`_ and `speech recognition <https://zh.wikipedia.org/wiki/speech recognition>`_ .  Models such as the `text classification model <https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleNLP/text_classification/nets.py>`_
`TagSpace <https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleRec/tagspace/train.py>`_ and `Multi-view Simnet <https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleRec/multiview_simnet/nets.py>`_.

For API Reference, please refer to :ref:`api_fluid_nets_sequence_conv_pool`


4.glu
-----
The full name:code:`glu` is Gated Linear Units, which is from the paper `Language Modeling with Gated Convolutional Networks <https://arxiv.org/pdf/1612.08083.pdf>`_. It consists by :ref:`api_fluid_layers_split` , :ref: `api_fluid_layers_sigmoid` and :ref:`api_fluid_layers_elementwise_mul`.
It divides the input data into 2 equal parts, and asks the second part for `Sigmoid <https://en.wikipedia.org/wiki/Sigmoid_function>`_ , then multiplies the first part data to get the output.

For API Reference, please refer to :ref:`api_fluid_nets_glu`


5.scaled_dot_product_attention
------------------------------
:code:`scaled_dot_product_attention` is from the paper `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_ , mainly composed of :ref:`api_fluid_layers_fc` and :ref:`api_fluid_layers_softmax` .
For the input data :code:`Queries` , :code:`Key` and :code:`Values`, calculate the :code:`Attention` according to the following formula.

.. math::
 Attention(Q, K, V)= softmax(QK^\mathrm{T})V

This module is widely used in the model of `machine translation <https://zh.wikipedia.org/zh/machine translation>`_, such as `Transformer <https://github.com/PaddlePaddle/models/tree/develop/Fluid/PaddleNLP/neural_machine_translation/transformer>`_ .

For API Reference, please refer to :ref:`api_fluid_nets_scaled_dot_product_attention`
