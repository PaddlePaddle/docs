.. _cn_api_fluid_io_get_program_parameter:

get_program_parameter
-------------------------------

.. py:function:: paddle.fluid.io.get_program_parameter(program)




该接口从Program中获取所有参数。

参数
::::::::::::

 - **program**  ( :ref:`cn_api_fluid_Program` ) – 从该Program中获取参数。

返回
::::::::::::
 包含此Program中所有参数的list

返回类型
::::::::::::
 list

代码示例
::::::::::::

COPY-FROM: paddle.fluid.io.get_program_parameter