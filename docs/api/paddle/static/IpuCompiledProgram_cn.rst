.. _cn_api_fluid_IpuCompiledProgram:

IpuCompiledProgram
-------------------------------


.. py:class:: paddle.static.IpuCompiledProgram(program, scope=None, ipu_strategy=None)


IpuCompiledProgram根据 `ipu_strategy` 的配置将输入的Program转换和优化成ipu所需要的形式，例如：前向图提取、计算图转化、无用的scale算子删除等。关于ipu_strategy更多信息。请参阅  ``paddle.static.IpuStrategy``。

参数
:::::::::
    - **program** (Program): 该参数为被执行的Program。如果没有设置此参数，则会使用默认的program, 即paddle.static.default_main_program()。
    - **scope** (Scope): 该参数表示执行当前program所使用的作用域。如果没有指定scope，将使用全局scope，即paddle.static.global_scope()。
    - **ipu_strategy** (IpuStrategy): 通过配置ipu_strategy，对计算图进行转换和优化，例如：计算图的float16模式、是否是训练模式、计算图需要用几个IPU等。默认为None。

返回
:::::::::
IpuCompiledProgram，初始化后的 ``IpuCompiledProgram`` 对象

代码示例
::::::::::

COPY-FROM: paddle.static.IpuCompiledProgram

.. py:method:: compile(self, feed_list, fetch_list)

该接口用于将Program进行编译，以便在ipu上运行。用户可以通过 `feed_list` 、`fetch_list` 传入计算图输入和输出的名字。

参数
:::::::::
    - **feed_list** （list）: 该参数为模型的输入变量的名字。
    - **fetch_list** （list）:  模型运行之后需要返回的变量的名字。

返回
:::::::::
Program，编译之后的 ``Program`` 对象


代码示例
:::::::::

COPY-FROM: paddle.static.IpuCompiledProgram.compile

