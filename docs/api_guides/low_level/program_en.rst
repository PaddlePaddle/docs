.. _api_guide_Program_en:

###############
Basic Concept
###############

==================
IR
==================

:code:`Paddle` represents the computation graph using an IR (Intermediate Representation) and leverages compiler principles, techniques, and tools to perform automatic optimization and code generation for neural networks.

The new IR represents structured control flow through a recursive nesting of :code:`Operation`, :code:`Region`, and :code:`Block`.

* An :code:`Operation` contains zero or more :code:`Regions`.
* A :code:`Region` contains zero or more :code:`Blocks`.
* A :code:`Block` contains zero or more :code:`Operations`.

These three components are recursively nested to describe complex model structures.

==================
Program
==================

A :code:`Program` represents a specific model. It consists of two parts: the computation graph and the weights. The model is equivalent to a directed acyclic graph (DAG), where :code:`Operation` serves as the nodes and :code:`Value` represents the edges.

:code:`Weight` is used to store the model's weight parameters separately, while :code:`Value` and :code:`Operation` abstract the computation graph.

:code:`Operation` represents a node in the computation graph. Each :code:`Operation` corresponds to an operator and contains zero or more :code:`Regions`.

:code:`Region` acts as a closure and contains zero or more :code:`Blocks`.

:code:`Block` represents a basic block conforming to SSA (Static Single Assignment) form and contains zero or more :code:`Operations`.

:code:`Value` represents a directed edge in the computation graph, linking two :code:`Operations` and describing the UD (Use-Define) chain in the program.

In a :code:`Program`, ``ModuleOp module_`` stores the computation graph, while the ``ParameterMap parameters_`` stores the weights. In the ``ModuleOp`` class, a :code:`Block` is used to store the contents of the computation graph.

.. _api_guide_Region_en:

=========
Region
=========

A :code:`Region` contains a list of :code:`Blocks`. The first :code:`Block` (if it exists) is referred to as the entry block of that :code:`Region`.

Unlike basic blocks, a key constraint of a :code:`Region` is that any :code:`Value` defined within the :code:`Region` can only be used inside that :code:`Region` and cannot be accessed externally.

When control flow enters a :code:`Region`, it effectively creates a new sub-scope. Upon exiting the :code:`Region`, all variables defined within this sub-scope can be reclaimed.

Control flow always enters a :code:`Region` through its entry block. Therefore, the parameters of a :code:`Region` can be described using the entry block's parameters without additional handling.

Once a :code:`Region` completes its execution and control flow returns from a child :code:`Block` to the :code:`Region`, there are two possible outcomes:

* The control flow enters another :code:`Region` of the same Op (which may be itself).
* The control flow returns to the parent Op of the :code:`Region`, marking the completion of one execution cycle of that Op.

The specific destination is determined by the semantics of the parent Op of the :code:`Region`.

Note: Before introducing control flow, an :code:`Operation` consists of its inputs, outputs, attributes, and type information. After incorporating control flow, an :code:`Operation` additionally includes its inputs (:code:`OpOperand`), outputs (:code:`OpResult`), attributes (:code:`AttributeMap`), successor blocks (:code:`BlockOperand`), and :code:`Region`. The successor blocks and :code:`Region` are newly added components.

.. _api_guide_Block_en:

=========
Block
=========

A :code:`Block` is equivalent to a basic block and contains a list of operators (``std::list<Operation*>``) that represent the computation semantics of the basic block.

When the last operator in a :code:`Block` finishes execution, the control flow follows one of two paths based on the semantics of the last operator (terminator operator) in the block:

* It transitions to another :code:`Block` within the same :code:`Region`. This :code:`Block` must be a successor block of the terminator operator.
* It returns to the parent :code:`Region` of the :code:`Block`, indicating the completion of one execution cycle of that :code:`Region`.

.. _api_guide_Operation_en:

=============
Operation
=============

An :code:`Operation` is a node in a directed graph. The information of an :code:`Operation` is divided into four parts: inputs (:code:`OpOperandImpl`), outputs (:code:`OpResultImpl`), attributes (:code:`Attribute`), and type information (:code:`OpInfo`). The number of inputs and outputs is determined at the time of construction and remains unchanged afterward.

:code:`Attribute` is used to describe an attribute. Users can temporarily store some runtime attributes within an operator, but these runtime attributes are only for assisting computation and are not allowed to alter the computation semantics. When exporting a model, all runtime attributes are removed by default.

:code:`Operation` type information (:code:`OpInfo`) is essentially an abstraction of the common properties shared by operators of the same type.


.. _api_guide_Weight_en:

=============
Weight
=============

:code:`Weight` attributes are a special type of attribute, typically involving a large amount of data. :code:`Paddle` stores weights separately and retrieves or saves weight values in the model using weight names. Currently, all model weights in :code:`Paddle` are of type ``Variable``.

=============
Operator
=============

In Paddle, all operations on data are represented by :code:`Operators`. Each :code:`Operator` performs a specific function, such as matrix multiplication, convolution, activation functions, etc. By combining these :code:`Operators`, complex computation graphs can be constructed to implement the forward and backward propagation of a model.

=========
Variable
=========

In Paddle, a :code:`Variable` can contain any type of value — most commonly a :code:`Tensor`.

All learnable parameters in the model are stored as :code:`Variable` objects in memory. In most cases, you don't need to manually create the learnable parameters in the network, as Paddle provides wrappers for almost all common neural network basic computation modules. For example, in the simplest fully connected model in a static graph, calling :code:`paddle.static.nn.fc` will automatically create the learnable parameters for the fully connected layer: connection weights (W) and biases (bias), without the need to explicitly call the :code:`variable` interface to create learnable parameters.

.. _api_guide_Name_en:

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

.. _api_guide_ParamAttr_en:

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
