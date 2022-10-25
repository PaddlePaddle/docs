.. _cn_api_fluid_io_set_program_state:

set_program_state
-------------------------------

.. py:function:: paddle.static.set_program_state(program, state_dict)


利用 ``state_dict`` 设置 ``Program`` 的参数和优化器信息。

如果参数的 shape 或 dtype 不匹配，则会引发异常。

.. note::
必须在运行 start_up_program 之后调用此函数。

参数
::::::::::::

    - **program** (Program) - 需要被设置的 ``Program`` 。
    - **state_dict** (dict) - 存储参数和优化器信息的 dict；dict 中 key 的类型为 Tensor 的名称，value 为 np.ndarray 类型的数据。

返回
::::::::::::
无。

代码示例
::::::::::::

COPY-FROM: paddle.static.set_program_state
