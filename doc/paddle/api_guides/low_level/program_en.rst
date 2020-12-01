.. _api_guide_Program_en:

###############
Basic Concept
###############

==================
Program
==================

:code:`Fluid` describes neural network configuration in the form of abstract grammar tree similar to that of a programming language, and the user's description of computation will be written into a Program. Program in Fluid replaces the concept of models in traditional frameworks. It can describe any complex model through three execution structures: sequential execution, conditional selection and loop execution. Writing :code:`Program` is very close to writing a common program. If you have tried programming before, you will naturally apply your expertise to it.

In brief：

* A model is a Fluid :code:`Program`  and can contain more than one :code:`Program` ;

* :code:`Program` consists of nested :code:`Block` , and the concept of :code:`Block` can be analogized to a pair of braces in C++ or Java, or an indentation block in Python.


* Computing in :code:`Block` is composed of three ways: sequential execution, conditional selection or loop execution, which constitutes complex computational logic.


* :code:`Block` contains descriptions of computation and computational objects. The description of computation is called Operator; the object of computation (or the input and output of Operator) is unified as Tensor. In Fluid, Tensor is represented by 0-leveled `LoD-Tensor <http://paddlepaddle.org/documentation/docs/zh/1.2/user_guides/howto/prepare_data/lod_tensor.html#permalink-4-lod-tensor>`_ .

.. _api_guide_Block_en:

=========
Block
=========

:code:`Block` is the concept of variable scope in advanced languages. In programming languages, Block is a pair of braces, which contains local variable definitions and a series of instructions or operators. Control flow structures :code:`if-else` and :code:`for` in programming languages can be equivalent to the following counterparts in deep learning:

+----------------------+-------------------------+
| programming languages| Fluid                   |
+======================+=========================+
| for, while loop      | RNN,WhileOP             |
+----------------------+-------------------------+
| if-else, switch      | IfElseOp, SwitchOp      |
+----------------------+-------------------------+
| execute sequentially | a series of layers      |
+----------------------+-------------------------+

As mentioned above,  :code:`Block` in Fluid describes a set of Operators that include sequential execution, conditional selection or loop execution, and the operating object of Operator: Tensor.



=============
Operator
=============

In Fluid, all operations of data are represented by :code:`Operator` . In Python, :code:`Operator` in Fluid is encapsulated into modules like :code:`paddle.fluid.layers` , :code:`paddle.fluid.nets` .

This is because some common operations on Tensor may consist of more basic operations. For simplicity, some encapsulation of the basic Operator is carried out inside the framework, including the creation of learnable parameters relied by an Operator, the initialization details of learnable parameters, and so on, so as to reduce the cost of further development.



More information can be read for reference. `Fluid Design Idea <../../advanced_usage/design_idea/fluid_design_idea.html>`_

.. _api_guide_Variable_en:

=========
Variable
=========

In Fluid， :code:`Variable` can contain any type of value -- in most cases a LoD-Tensor.

All the learnable parameters in the model are kept in the memory space in form of :code:`Variable` . In most cases, you do not need to create the learnable parameters in the network by yourself. Fluid provides encapsulation for almost common basic computing modules of the neural network. Taking the simplest full connection model as an example, calling :code:`fluid.layers.fc` directly creates two learnable parameters for the full connection layer, namely, connection weight (W) and bias, without explicitly calling :code:`Variable` related interfaces to create learnable parameters.

.. _api_guide_Name:

=========
Name
=========

In Fluid, some layers contain the parameter :code:`name` , such as :ref:`api_fluid_layers_fc` . This :code:`name` is generally used as the prefix identification of output and weight in network layers. The specific rules are as follows:

* Prefix identification for output of layers. If :code:`name` is specified in the layer, Fluid will name the output with ``nameValue.tmp_number`` . If the :code:`name` is not specified, ``OPName_number.tmp_number`` is automatically generated to name the layer. The numbers are automatically incremented to distinguish different network layers under the same operator.

* Prefix identification for weight or bias variable. If the weight and bias variables are created by ``param_attr`` and ``bias_attr`` in operator, such as :ref:`api_fluid_layers_embedding` 、 :ref:`api_fluid_layers_fc` , Fluid will generate ``prefix.w_number`` or ``prefix.b_number`` as unique identifier to name them, where the ``prefix`` is :code:`name` specified by users or ``OPName_number`` generated by default. If :code:`name` is specified in ``param_attr`` and ``bias_attr`` , the :code:`name` is no longer generated automatically. Refer to the sample code for details.

In addition, the weights of multiple network layers can be shared by specifying the :code:`name` parameter in :ref:`api_fluid_ParamAttr`.

Sample Code:

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=1)
    emb = fluid.layers.embedding(input=x, size=(128, 100))  # embedding_0.w_0
    emb = fluid.layers.Print(emb) # Tensor[embedding_0.tmp_0]

    # default name
    fc_none = fluid.layers.fc(input=emb, size=1)  # fc_0.w_0, fc_0.b_0
    fc_none = fluid.layers.Print(fc_none)  # Tensor[fc_0.tmp_1]

    fc_none1 = fluid.layers.fc(input=emb, size=1)  # fc_1.w_0, fc_1.b_0
    fc_none1 = fluid.layers.Print(fc_none1)  # Tensor[fc_1.tmp_1]

    # name in ParamAttr
    w_param_attrs = fluid.ParamAttr(name="fc_weight", learning_rate=0.5, trainable=True)
    print(w_param_attrs.name)  # fc_weight

    # name == 'my_fc'
    my_fc1 = fluid.layers.fc(input=emb, size=1, name='my_fc', param_attr=w_param_attrs) # fc_weight, my_fc.b_0
    my_fc1 = fluid.layers.Print(my_fc1)  # Tensor[my_fc.tmp_1]

    my_fc2 = fluid.layers.fc(input=emb, size=1, name='my_fc', param_attr=w_param_attrs) # fc_weight, my_fc.b_1
    my_fc2 = fluid.layers.Print(my_fc2)  # Tensor[my_fc.tmp_3]

    place = fluid.CPUPlace()
    x_data = np.array([[1],[2],[3]]).astype("int64")
    x_lodTensor = fluid.create_lod_tensor(x_data, [[1, 2]], place)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    ret = exe.run(feed={'x': x_lodTensor}, fetch_list=[fc_none, fc_none1, my_fc1, my_fc2], return_numpy=False)


In the above example, ``fc_none`` and ``fc_none1`` are not specified :code:`name` parameter, so this two layers are named with ``fc_0.tmp_1`` and ``fc_1.tmp_1`` in the form ``OPName_number.tmp_number`` , where the numbers in ``fc_0`` and ``fc_1`` are automatically incremented to distinguish this two fully connected layers. The other two fully connected layers ``my_fc1`` and ``my_fc2`` both specify the :code:`name` parameter with same values. Fluid will distinguish the two layers by suffix ``tmp_number`` . That is ``my_fc.tmp_1`` and ``my_fc.tmp_3`` .

Variables created in ``emb`` layer and ``fc_none`` , ``fc_none1`` are named by the ``OPName_number`` , such as ``embedding_0.w_0`` 、 ``fc_0.w_0`` 、 ``fc_0.b_0`` . And the prefix is consistent with the prefix of network layer. The ``my_fc1`` layer and ``my_fc2`` layer preferentially name the shared weight with ``fc_weight`` specified in ``ParamAttr`` . The bias variables ``my_fc.b_0`` and ``my_fc.b_1`` are identified suboptimally with :code:`name` int the operator as prefix.

In the above example, the ``my_fc1`` and ``my_fc2`` two fully connected layers implement the sharing of weight parameters by constructing ``ParamAttr`` and specifying the :code:`name` parameter.

.. _api_guide_ParamAttr:

=========
ParamAttr
=========

==================
Related API
==================


* A single neural network configured by the user is called :ref:`api_fluid_Program` . It is noteworthy that when training neural networks, users often need to configure and operate multiple :code:`Program` . For example,  :code:`Program` for parameter initialization, :code:`Program` for training,  :code:`Program` for testing, etc.


* Users can also use :ref:`api_fluid_program_guard` with :code:`with` to modify the configured :ref:`api_fluid_default_startup_program` and :ref:`api_fluid_default_main_program` .


* In Fluid，the execution order in a Block is determined by control flow，such as :ref:`api_fluid_layers_IfElse` , :ref:`api_fluid_layers_While` and :ref:`api_fluid_layers_Switch` . For more information, please refer to： :ref:`api_guide_control_flow_en`
