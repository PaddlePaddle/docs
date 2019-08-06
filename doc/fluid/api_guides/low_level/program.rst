.. _api_guide_Program:

#########
基础概念
#########

==================
Program
==================

:code:`Fluid` 中使用类似于编程语言的抽象语法树的形式描述用户的神经网络配置，用户对计算的描述都将写入一段Program。Fluid 中的 Program 替代了传统框架中模型的概念，通过对顺序执行、条件选择和循环执行三种执行结构的支持，做到对任意复杂模型的描述。书写 :code:`Program` 的过程非常接近于写一段通用程序，如果您已经具有一定的编程经验，会很自然地将自己的知识迁移过来。


总得来说：

* 一个模型是一个 Fluid :code:`Program` ,一个模型可以含有多于一个 :code:`Program` ；

* :code:`Program` 由嵌套的 :code:`Block` 构成，:code:`Block` 的概念可以类比到 C++ 或是 Java 中的一对大括号，或是 Python 语言中的一个缩进块；

* :code:`Block` 中的计算由顺序执行、条件选择或者循环执行三种方式组合，构成复杂的计算逻辑；

* :code:`Block` 中包含对计算和计算对象的描述。计算的描述称之为 Operator；计算作用的对象（或者说 Operator 的输入和输出）被统一为 Tensor，在Fluid中，Tensor 用层级为0的 :ref:`cn_user_guide_Lod\_Tensor` 表示。




=========
Block
=========

:code:`Block` 是高级语言中变量作用域的概念，在编程语言中，Block是一对大括号，其中包含局部变量定义和一系列指令或操作符。编程语言中的控制流结构 :code:`if-else` 和 :code:`for` 在深度学习中可以被等效为：

+----------------------+-------------------------+
| 编程语言              | Fluid                   |
+======================+=========================+
| for, while loop      | RNN,WhileOP             |
+----------------------+-------------------------+
| if-else, switch      | IfElseOp, SwitchOp      |
+----------------------+-------------------------+
| 顺序执行              | 一系列 layers            |
+----------------------+-------------------------+

如上文所说，Fluid 中的 :code:`Block` 描述了一组以顺序、选择或是循环执行的 Operator 以及 Operator 操作的对象：Tensor。




=============
Operator
=============

在 Fluid 中，所有对数据的操作都由 :code:`Operator` 表示，为了便于用户使用，在 Python 端，Fluid 中的 :code:`Operator` 被一步封装入 :code:`paddle.fluid.layers` ， :code:`paddle.fluid.nets` 等模块。

这是因为一些常见的对 Tensor 的操作可能是由更多基础操作构成，为了提高使用的便利性，框架内部对基础 Operator 进行了一些封装，包括创建 Operator 依赖可学习参数，可学习参数的初始化细节等，减少用户重复开发的成本。


更多内容可参考阅读 `Fluid设计思想 <../../advanced_usage/design_idea/fluid_design_idea.html>`_


=========
Variable
=========

Fluid 中的 :code:`Variable` 可以包含任何类型的值———在大多数情况下是一个 :ref:`cn_user_guide_Lod\_Tensor` 。

模型中所有的可学习参数都以 :code:`Variable` 的形式保留在内存空间中，您在绝大多数情况下都不需要自己来创建网络中的可学习参数， Fluid 为几乎常见的神经网络基本计算模块都提供了封装。以最简单的全连接模型为例，调用 :code:`fluid.layers.fc` 会直接为全连接层创建连接权值( W )和偏置（ bias ）两个可学习参数，无需显示地调用 :code:`variable` 相关接口创建可学习参数。


=========
相关API
=========

* 用户配置的单个神经网络叫做 :ref:`cn_api_fluid_Program` 。值得注意的是，训练神经网
  络时，用户经常需要配置和操作多个 :code:`Program` 。比如参数初始化的
  :code:`Program` ， 训练用的 :code:`Program` ，测试用的
  :code:`Program` 等等。


* 用户还可以使用 :ref:`cn_api_fluid_program_guard` 配合 :code:`with` 语句，修改配置好的 :ref:`cn_api_fluid_default_startup_program` 和 :ref:`cn_api_fluid_default_main_program` 。

* 在Fluid中，Block内部执行顺序由控制流决定，如 :ref:`cn_api_fluid_layers_IfElse` , :ref:`cn_api_fluid_layers_While`, :ref:`cn_api_fluid_layers_Switch` 等，更多内容可参考： :ref:`api_guide_control_flow`
