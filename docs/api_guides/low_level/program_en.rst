.. _api_guide_Program_en:

###############
Basic Concept
###############

==================
Program
==================

In PaddlePaddle, a Program is a static graph model, similar to programs in other programming languages. Static graph programming follows a "define-and-run" approach:

* Define: The complete neural network architecture is predefined in the code.

* Compile: PaddlePaddle represents the neural network as a Program data structure and performs compilation optimizations.

* Execute: An executor is invoked to obtain the computation results.

This approach allows for efficient execution but requires the entire network structure to be defined before running the program.

* A :code:`Program` consists of nested :code:`Blocks`. The concept of a :code:`Block` can be likened to a pair of curly braces ``{}`` in languages like C++ or Java, or to an indented block in Python.

* The computation in the :code:`Block` is composed of three types of execution: sequential execution, conditional selection, and loop execution, which together form a complex computational logic.

* The :code:`Block` contains descriptions of the computation and the objects involved in the computation. The description of the computation is called the :code:`Operator`; the objects on which the computation acts (or the inputs and outputs of the :code:`Operator`) are unified as :code:`Tensors`.

.. _api_guide_Block_en:

=========
Block
=========

The :code:`Block` is the concept of variable scope in high-level languages, similar to a pair of curly braces in C or Java, which contain local variable definitions and a series of instructions or operators.

The :code:`Block` is the fundamental unit in a computation graph used to represent computational logic. It contains a series of operations (:code:`Operator`) and computational objects (:code:`Tensor`), supporting control structures such as sequential execution, conditional selection, and loop execution, thereby building complex computational flows.

* Computation description: The :code:`Block` contains multiple :code:`Operators` internally, with each :code:`Operator` representing a computational operation, such as addition, convolution, etc.

* Object description: The computational objects in the :code:`Block` are unified as :code:`Tensors`, representing multi-dimensional arrays or matrices, and are the basic units of data storage and transmission.

* Control structures: The :code:`Block` supports control structures such as sequential execution, conditional selection, and loop execution, making the computational flow more flexible and complex.

In the PaddlePaddle computation graph, :code:`Block`, :code:`Operator`, and :code:`Tensor` together form the backbone of the computational flow. The :code:`Block` provides a container function, organizing and managing the internal :code:`Operators` and :code:`Tensors`, thereby enabling efficient construction and execution of the computation graph.

=============
Operator
=============

In Paddle, all operations on data are represented by :code:`Operators`. Each :code:`Operator` performs a specific function, such as matrix multiplication, convolution, activation functions, etc. By combining these :code:`Operators`, complex computation graphs can be constructed to implement the forward and backward propagation of a model.

=========
Variable
=========

In Paddle, a :code:`Variable` can contain any type of value — most commonly a :code:`Tensor`.

All learnable parameters in the model are stored as :code:`Variable` objects in memory. In most cases, you don't need to manually create the learnable parameters in the network, as Paddle provides wrappers for almost all common neural network basic computation modules. For example, in the simplest fully connected model in a static graph, calling :code:`paddle.static.nn.fc` will automatically create the learnable parameters for the fully connected layer: connection weights (W) and biases (bias), without the need to explicitly call the :code:`variable` interface to create learnable parameters.

.. _api_guide_Name:

=========
Name
=========

In Paddle, some network layers include a :code:`name` parameter, such as in the :code:`paddle.static.nn.fc` API. This :code:`name` is generally used as a prefix identifier for the network layer's output and weights. The specific rules are as follows:

* The prefix identifier used for the network layer output. If the :code:`name` parameter is specified in the network layer, Paddle will use the :code:`name` value followed by ``.tmp_number`` as a unique identifier for naming the network layer's output. If the :code:`name` parameter is not specified, it will use the format ``OP_name_number.tmp_number`` for naming, where the numbers will automatically increment to distinguish different network layers under the same OP name.

* The prefix identifier used for weight or bias variables. If weight or bias variables are created in the network layer through ``param_attr`` and ``bias_attr``, such as in the :ref:`api_nn_embedding` or :ref:`api_static_nn_fc` APIs, Paddle will automatically generate a unique identifier in the format ``prefix.w_number`` or ``prefix.b_number`` for naming them, where ``prefix`` is either the user-specified :code:`name` or the automatically generated ``OP_name_number``. If a :code:`name` is specified in ``param_attr`` or ``bias_attr``, this :code:`name` will be used, and the automatic generation will not occur. For details, please refer to the example code.

Additionally, in the :ref:`api_ParamAttr` API, you can achieve weight sharing across multiple network layers by specifying the :code:`name` parameter.

Sample Code:

.. code-block:: python

    import paddle
    import numpy as np

    embedding = paddle.nn.Embedding(num_embeddings=128, embedding_dim=100)
    emb = embedding(x)  # embedding_0.w_0
    print(emb) # Tensor[embedding_0.tmp_0]

    # default name
    fc = paddle.nn.Linear(in_features=100, out_features=1)
    fc_out = fc(emb)  # fc_0.w_0, fc_0.b_0
    print(fc_out)  # Tensor[fc_0.tmp_1]

    fc1 = paddle.nn.Linear(in_features=100, out_features=1)  # fc_1.w_0, fc_1.b_0
    fc1_out = fc1(emb)  # fc_1.w_0, fc_1.b_0
    print(fc1_out)  # Tensor[fc_1.tmp_1]

    # name in ParamAttr
    w_param_attrs = paddle.ParamAttr(name="fc_weight", learning_rate=0.5, trainable=True)
    print(w_param_attrs.name)  # fc_weight

    # name == 'my_fc'
    my_fc = paddle.nn.Linear(in_features=100, out_features=1, name='my_fc', weight_attr=w_param_attrs)
    my_fc_out = my_fc(emb) # fc_weight, my_fc.b_0
    print(my_fc_out)  # Tensor[my_fc.tmp_1]

    my_fc2 = paddle.nn.Linear(in_features=100, out_features=1, name='my_fc', weight_attr=w_param_attrs)
    my_fc2_out = my_fc2(emb) # fc_weight, my_fc.b_1
    print(my_fc2_out)  # Tensor[my_fc.tmp_3]

    place = paddle.CPUPlace()

    exe = paddle.static.Executor(place)

    exe.run(paddle.static.default_startup_program())

    ret = exe.run(feed={'x': x}, fetch_list=[fc_out, fc1_out, my_fc_out, my_fc2_out], return_numpy=False)


In the above example, ``fc_none`` and ``fc_none1`` did not specify the :code:`name` parameter, so the outputs of these OPs are named using the format ``OP_name_number.tmp_number``: ``fc_0.tmp_1`` and ``fc_1.tmp_1``, where the numbers in ``fc_0`` and ``fc_1`` automatically increment to distinguish the two fully connected layers. ``my_fc1`` and ``my_fc2`` both specified the :code:`name` parameter, but with the same value. Paddle differentiates them by appending ``tmp_number``, resulting in ``my_fc.tmp_1`` and ``my_fc.tmp_3``.

For variables created in the network layers, the ``emb`` layer, ``fc_none``, and ``fc_none1`` layers default to naming weight or bias variables with the prefix ``OP_name_number``, such as ``embedding_0.w_0``, ``fc_0.w_0``, and ``fc_0.b_0``, with the prefix matching the OP output. The ``my_fc1`` and ``my_fc2`` layers prioritize the ``fc_weight`` specified in ``ParamAttr`` as the name for the shared weights. The bias variables ``my_fc.b_0`` and ``my_fc.b_1`` are next in priority, named with the :code:`name` prefix.

In the above example, the two fully connected layers, ``my_fc1`` and ``my_fc2``, achieved weight variable sharing by constructing ``ParamAttr`` and specifying the :code:`name` parameter.

.. _api_guide_ParamAttr:

=========
ParamAttr
=========

``ParamAttr`` is a configuration class used to set the attributes of model parameters, such as weights and biases. Through ``ParamAttr``, users can flexibly define characteristics such as parameter initialization methods, regularization strategies, gradient clipping, and model averaging.

Sample Code:

.. code-block:: python
    import paddle
    from paddle import ParamAttr

    # Create a fully connected layer and set the attributes for the weights and biases.
    fc = paddle.nn.Linear(in_features=128, out_features=64,
                          weight_attr=ParamAttr(
                              name='fc_weight',
                              initializer=paddle.nn.initializer.XavierUniform(),
                              regularizer=paddle.regularizer.L2Decay(0.0001)
                          ),
                           bias_attr=ParamAttr(
                              name='fc_bias',
                              initializer=paddle.nn.initializer.Constant(0.0)
                          ))

In the above example, ``weight_attr`` and ``bias_attr`` set the attributes for the weights and biases, respectively. The :code:`name` specifies the name of the parameter. The ``initializer`` sets the initialization method for the parameter, and the ``regularizer`` sets the regularization strategy for the parameter.

==================
Related API
==================


* The user-configured individual neural network is called a :code:`Program`. It is important to note that during the training of a neural network, users often need to configure and operate multiple :code:`Programs`. For example, a :code:`Program` for parameter initialization, a :code:`Program` for training, and a :code:`Program` for testing, etc.


* Users can also use the :ref:`api_program_guard` in conjunction with the :code:`with` statement to modify the configured :ref:`api_default_startup_program` and :ref:`api_default_main_program`.
