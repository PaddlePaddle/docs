.. _api_guide_Program_en:

###############################
Program/Block/Operator/Variable
###############################

==================
Program
==================

:code:`Fluid` describes the user's neural network configuration in the form of abstract grammar tree similar to programming language, and the user's description of computation will be written into a Program. Program in Fluid replaces the concept of model in traditional framework. It can describe any complex model by supporting three execution structures: sequential execution, conditional selection and loop execution. Writing :code:`Program` is very close to writing a general program. If you have some programming experience, you will naturally transfer your knowledge.

In brief：

* A model is a Fluid :code:`Program`  and can contain more than one :code:`Program` ;

* :code:`Program` consists of nested :code:`Block` , and the concept of :code:`Block` can be analogized to a pair of braces in C++ or Java, or an indentation block in Python.


* Computing in :code:`Block` is composed of three ways: sequential execution, conditional selection or loop execution, which constitutes complex computational logic.


* :code:`Block` contains descriptions of computation and computational objects. The description of computation is called Operator; the object of computation (or the input and output of Operator) is unified as Tensor. In Fluid, Tensor is represented by 0-leveled `LoD-Tensor <http://paddlepaddle.org/documentation/docs/zh/1.2/user_guides/howto/prepare_data/lod_tensor.html#permalink-4-lod-tensor>`_ .


=========
Block
=========

:code:`Block` is the concept of variable scope in advanced languages. In programming languages, Block is a pair of braces, which contains local variable definitions and a series of instructions or operators. Control flow structures :code:`if-else` and :code:`for` in programming languages can be equivalent to in deep learning:

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

In Fluid, all operations of data are represented by :code:`Operator` . In Python, :code:`Operator` in Fluid is encapsulated into modules such as: :code:`paddle.fluid.layers` , :code:`paddle.fluid.nets` , and so on.

This is because some common operations on Tensor may consist of more basic operations. In order to improve the convenience of use, some encapsulation of the basic Operator is carried out inside the framework, including the creation of Operator dependent on learnable parameters, the initialization details of learnable parameters, and so on, so as to reduce the cost of repeating development by users.



More information can be read for reference. `Fluid Design Idea <../../advanced_usage/design_idea/fluid_design_idea.html>`_ 


=========
Variable
=========

In Fluid， :code:`Variable` can contain any type of value -- in most cases a LoD-Tensor.

All the learnable parameters in the model are kept in the memory space in form of :code:`Variable` . In most cases, you do not need to create the learnable parameters in the network by yourself. Fluid provides encapsulation for almost common basic computing modules of the neural network. Taking the simplest full connection model as an example, calling :code:`fluid.layers.fc` directly creates two learnable parameters for the full connection layer, namely, connection weight (W) and bias, without explicitly calling :code:`variable` related interfaces to create learnable parameters.

=========
Related API
=========


* A single neural network for user configuration is called :ref:`api_fluid_Program` . It is noteworthy that when training neural networks, users often need to configure and operate multiple :code:`Program` . For example,  :code:`Program` for parameter initialization, :code:`Program` for training,  :code:`Program` for testing, etc.


* Users can also use :ref:`api_fluid_program_guard` with :code:`with` to modify the configured :ref:`api_fluid_default_startup_program` and :ref:`api_fluid_default_main_program` .


* In Fluid，Block internal execution order is determined by control flow，such as :ref:`api_fluid_layers_IfElse` , :ref:`api_fluid_layers_While` and :ref:`api_fluid_layers_Switch` . For more information, please reference to： :ref:`api_guide_control_flow_en` 
