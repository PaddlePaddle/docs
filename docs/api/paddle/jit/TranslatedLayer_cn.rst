.. _cn_api_fluid_dygraph_TranslatedLayer:

TranslatedLayer
-------------------------------

.. py:class:: paddle.jit.TranslatedLayer(programs, persistable_vars)

``TranslatedLayer`` 是一个命令式编程模式 :ref:`cn_api_fluid_dygraph_Layer` 的继承类，
通过 :ref:`cn_api_paddle_jit_load` 载入构建。能够像一般 ``Layer`` 一样在 train 或者 eval 模式下使用。

.. note::
  ``TranslatedLayer`` 对象不能够通过构造函数创建，仅能够通过 :ref:`cn_api_paddle_jit_load` 接口载入构建。

代码示例
::::::::::::
COPY-FROM: paddle.jit.translated_layer.TranslatedLayer

方法
::::::::::::
program(method_name='forward'):
'''''''''

获取 TranslatedLayer 中指定方法对应的 Program。

**参数**

    - **method_name** (string) - 要获取的 Porgram 对应的方法名。默认值为"forward"。

**返回**
Program

**代码示例**
COPY-FROM: paddle.jit.translated_layer.TranslatedLayer.program
