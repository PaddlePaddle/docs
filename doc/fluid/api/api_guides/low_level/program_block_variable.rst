###############################
Program/Block/Operator/Variable
###############################

:code:`Fluid` 中使用类似于编程语言的抽象语法树的形式描述用户的经网络配置。直接操
作 `Operator`, `Variable` 虽然未被禁止，但十分不推荐。推荐您使用
:code:`fluid.layers` 中的相关API配置网络。

其中

* 用户配置的单个神经网络叫做 :ref:`api_fluid_Program` 。值得注意的是，训练神经网
  络时，用户经常需要配置和操作多个 :ref:`api_fluid_Program` 。比如参数初始化的
  :ref:`api_fluid_Program` ， 训练用的 :ref:`api_fluid_Program` ，测试用的
  :ref:`api_fluid_Program` 等等。

* :ref:`api_fluid_Program` 中包含 :code:`Block` 。 :code:`Block` 是高级语言中变量
  作用域的概念。

* :code:`Operator` 和 :code:`Variable` 。 :code:`Variable` 表示一段内存空间，
  :code:`Operator` 表示相应的操作。
