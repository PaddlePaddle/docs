.. _api_guide_Program:

###############################
Program/Block/Operator/Variable
###############################


:code:`Fluid` 中使用类似于编程语言的抽象语法树的形式描述用户的神经网络配置。用户对计算的描述都将写入一段Program。

一段 Program 中包含 :code:`Block`、:code:`Operator` 和 :code:`Variable`。

嵌套的 :code:`Block` 是 :code:`Program` 中的基本结构，它们之间的关系可以被表示为：

- Program： 
	- 一些嵌套的 :code:`Block`:
		- 一些 ``Variable`` 定义
		- 一系列的 ``Operator`` 

:code:`Block` 是高级语言中变量作用域的概念。在 Fluid 中，当执行到一个 :code:`Block` 时，框架会添加一个新的作用域，实现 :code:`Block` 里定义的 Variable 和 Operator。

:code:`Variable` 表示一段内存空间，里面包含任何类型的值———在大多数情况下是一个张量，但在RNN情况下也可以是一些整数id或其他变量的作用域。

:code:`Operator` 表示对 :code:`Variable` 进行的一系列操作。

更多内容可参考阅读 `Fluid设计思想 <../../../advanced_usage/design_idea/fluid_design_idea.html>`_ 

**提示：**

直接操作 ``Operator``, ``Variable`` 虽然未被禁止，但十分不推荐。推荐您使用 :code:`fluid.layers` 中的相关API配置网络。

**相关API：**

* 用户配置的单个神经网络叫做 :ref:`cn_api_fluid_Program` 。值得注意的是，训练神经网
  络时，用户经常需要配置和操作多个 :ref:`cn_api_fluid_Program` 。比如参数初始化的
  :ref:`cn_api_fluid_program` ， 训练用的 :ref:`cn_api_fluid_program` ，测试用的
  :ref:`cn_api_fluid_Program` 等等。


* 用户还可以使用 :ref:`cn_api_fluid_program_guard` 配合 :code:`with` 语句，修改配置好的 :ref:`cn_api_fluid_default_startup_program` 和 :ref:`cn_api_fluid_default_main_program` 。



* 在Fluid中，Block内部执行顺序由控制流决定，如 :ref:`cn_api_fluid_layers_IfElse` , :ref:`cn_api_fluid_layers_While`, :ref:`cn_api_fluid_layers_Switch` 等。 相关内容可参考： :ref:`api_guide_control_flow` 
