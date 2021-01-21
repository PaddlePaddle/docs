.. _cn_api_fluid_default_main_program:

default_main_program
-------------------------------

.. py:function:: paddle.fluid.default_main_program()





此接口可以获取当前用于存储OP和Tensor描述信息的 ``default main program``。

``default main program`` 是许多编程接口中Program参数的默认值。例如对于 ``Executor.run()`` 如果用户没有传入Program参数，会默认使用 ``default main program`` 。

可以使用 :ref:`cn_api_fluid_program_guard` 来切换 ``default main program``。 

返回： :ref:`cn_api_fluid_Program` ，当前默认用于存储OP和Tensor描述的Program。


**代码示例**

.. code-block:: python

        import paddle

        paddle.enable_static()
        # Sample Network:
        x = paddle.static.data(name='x', shape=[100, 100], dtype='float32')
        y = paddle.static.data(name='x', shape=[100, 100], dtype='float32')
        out = paddle.add(x, y)

        #print the number of blocks in the program, 1 in this case
        print(paddle.static.default_main_program().num_blocks) # 1
        #print the default_main_program
        print(paddle.static.default_main_program())
