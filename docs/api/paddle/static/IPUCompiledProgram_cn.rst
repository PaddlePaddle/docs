.. _cn_api_fluid_IPUCompiledProgram:

IPUCompiledProgram
-------------------------------


.. py:class:: paddle.static.IPUCompiledProgram(program, scope=None, ipu_strategy=None)


IPUCompiledProgram根据 `ipu_strategy` 的配置将输入的Program转换和优化成ipu所需要的形式，例如：前向图提取、计算图转化、无用的sclae算子删除等。关于ipu_strategy更多信息。请参阅  ``fluid.IPUStrategy``。

参数
:::::::::
    - **program** (Program): 该参数为被执行的Program。
    - **scope** (Scope): 该参数表示执行当前program所使用的作用域。如果没有指定scope，将使用全局scope，即paddle.static.global_scope()。
    - **ipu_strategy** (IpuStrategy): 通过配置ipu_strategy，对计算图进行转换和优化，例如：计算图的float16模式、是否是训练模式、计算图需要用几个IPU等。默认为None。

返回
:::::::::
IPUCompiledProgram，初始化后的 ``IPUCompiledProgram`` 对象

代码示例
::::::::::

COPY-FROM: paddle.static.IPUCompiledProgram

.. py:method:: compile(self, feed_list, fetch_list)
该接口用于将Program进行编译，以便在ipu上运行该。用户可以通过 `feed_list` 、`fetch_list` 传入计算图输入和输出的描述。

参数
:::::::::
    - **feed_list** （list）: 该参数为模型的输入变量的名字。
    - **fetch_list** （list）:  模型运行之后需要返回的变量的名字。

返回
:::::::::
Program，编译之后的 ``Program`` 对象


代码示例
:::::::::

COPY-FROM: paddle.static.IPUCompiledProgram

