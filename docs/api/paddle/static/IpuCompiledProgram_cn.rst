.. _cn_api_fluid_IpuCompiledProgram:

IpuCompiledProgram
-------------------------------


.. py:class:: paddle.static.IpuCompiledProgram(program, scope=None, ipu_strategy=None)


IpuCompiledProgram将输入的Program转换和优化成IPU所需要的形式，例如：前向图提取、计算图转化、无用的scale算子删除等。

参数
:::::::::
    - **program** (Program，可选): 该参数为被执行的Program。默认值为None，表示将使用默认的program, 即paddle.static.default_main_program()。
    - **scope** (Scope，可选): 该参数表示执行当前program所使用的作用域。默认值为None，将使用全局scope，即paddle.static.global_scope()。
    - **ipu_strategy** (IpuStrategy，可选): 根据传入的ipu_strategy实例，对Program进行转换和优化，例如：计算图的float16模式、是否是训练模式、计算图需要用几个IPU等。默认为None，表示将使用默认的ipu_strategy转换Program。

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

