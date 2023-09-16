.. _api_guide_nets_en:

################
Complex Networks
################

When dealing with complex functions, we usually need to code a lot to build a complex `Neural Network <https://en.wikipedia.org/wiki/Artificial_neural_network>`_ .
Therefore, in order to make it easier for users to build complex network models, we provide some common basic function modules to simplify the user's code and reduce development costs.
These modules are usually composed of fine-grained functions combined based on certain logics. For implementation, please refer to `nets <https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/nets.py>`_ .

1.simple_img_conv_pool
----------------------

:code:`simple_img_conv_pool` is got by concatenating :ref:`api_fluid_layers_conv2d` with :ref:`api_fluid_layers_pool2d` .
This module is widely used in image classification models, such as the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_ number classification.

For API Reference, please refer to :ref:`api_fluid_nets_simple_img_conv_pool`


2.img_conv_group
----------------

:code:`img_conv_group` is composed of :ref:`api_fluid_layers_conv2d` , :ref:`api_fluid_layers_batch_norm`, :ref:`api_fluid_layers_dropout` and :ref:`api_fluid_layers_pool2d`.
This module can implement the combination of multiple :ref:`api_fluid_layers_conv2d` , :ref:`api_fluid_layers_batch_norm` , :ref:`api_fluid_layers_dropout` and a single :ref:`api_fluid_layers_pool2d`.
Among them, the number of :ref:`api_fluid_layers_conv2d` , :ref:`api_fluid_layers_batch_norm` and :ref:`api_fluid_layers_dropout` can be controlled separately, resulting in various combinations.
This module is widely used in more complex image classification tasks, such as `VGG <https://arxiv.org/pdf/1409.1556.pdf>`_.

For API Reference, please refer to :ref:`api_fluid_nets_img_conv_group`


3.sequence_conv_pool
--------------------

:code:`sequence_conv_pool` is got by concatenating :ref:`api_fluid_layers_sequence_conv` with :ref:`api_fluid_layers_sequence_pool`.
The module is widely used in the field of `natural language processing <https://en.wikipedia.org/wiki/Natural_language_processing>`_ and `speech recognition <https://en.wikipedia.org/wiki/Speech_recognition>`_ .  Models such as the `text classification model <https://github.com/PaddlePaddle/models/blob/develop/PaddleNLP/text_classification/nets.py>`_ ,
`TagSpace <https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/tagspace/train.py>`_ and `Multi view Simnet <https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/multiview_simnet/nets.py>`_.

For API Reference, please refer to :ref:`api_fluid_nets_sequence_conv_pool`


4.glu
-----
The full name of :code:`glu` is Gated Linear Units, which originates from the paper `Language Modeling with Gated Convolutional Networks <https://arxiv.org/pdf/1612.08083.pdf>`_ . It consists of :ref:`api_fluid_layers_split` , :ref:`api_fluid_layers_sigmoid` and :ref:`api_fluid_layers_elementwise_mul`.
It divides the input data into 2 equal parts, calculates the `Sigmoid <https://en.wikipedia.org/wiki/Sigmoid_function>`_ of second part, and then performs dot product of the sigmoid vlaue with the first part to get the output.

For API Reference, please refer to :ref:`api_fluid_nets_glu`


5.scaled_dot_product_attention
------------------------------
:code:`scaled_dot_product_attention` originates from the paper `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_ , mainly composed of :ref:`api_fluid_layers_fc` and :ref:`api_fluid_layers_softmax` .
For the input data :code:`Queries` , :code:`Key` and :code:`Values`, calculate the :code:`Attention` according to the following formula.

.. math::
 Attention(Q, K, V)= softmax(QK^\mathrm{T})V

This module is widely used in the model of `machine translation <https://en.wikipedia.org/wiki/Machine_translation>`_, such as `Transformer <https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/neural_machine_translation/transformer>`_ .

For API Reference, please refer to :ref:`api_fluid_nets_scaled_dot_product_attention`
