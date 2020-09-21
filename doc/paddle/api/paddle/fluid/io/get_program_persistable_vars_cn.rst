.. _cn_api_fluid_io_get_program_persistable_vars:

get_program_persistable_vars
-------------------------------

.. py:function:: paddle.fluid.io.get_program_persistable_vars(program)




该接口从Program中获取所有persistable的变量。

参数:
 - **program**  ( :ref:`cn_api_fluid_Program` ) – 从该Program中获取persistable的变量。

返回: 包含此Program中所有persistable的变量

返回类型: list

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.data(name="img", shape=[64, 784])
    w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
    b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
    list_para  = fluid.io.get_program_persistable_vars(  fluid.default_main_program() )

