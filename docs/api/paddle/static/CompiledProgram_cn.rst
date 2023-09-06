.. _cn_api_fluid_CompiledProgram:

CompiledProgram
-------------------------------


.. py:class:: paddle.static.CompiledProgram(program_or_graph, build_strategy=None)


CompiledProgram 根据 `build_strategy` 的配置将输入的 Program 或 Graph 进行转换和优化，例如：计算图中算子融合、计算图执行过程中开启内存/显存优化等，关于 build_strategy 更多信息。请参阅  ``fluid.BuildStrategy`` 。

参数
:::::::::
    - **program_or_graph** (Graph|Program)：该参数为被执行的 Program 或 Graph。
    - **build_strategy** (BuildStrategy)：通过配置 build_strategy，对计算图进行转换和优化，例如：计算图中算子融合、计算图执行过程中开启内存/显存优化等。关于 build_strategy 更多信息，请参阅  ``fluid.BuildStrategy``。默认为 None。

返回
:::::::::
CompiledProgram，初始化后的 ``CompiledProgram`` 对象。

代码示例
::::::::::

COPY-FROM: paddle.static.CompiledProgram
