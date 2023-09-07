.. _cn_api_paddle_static_nn_while_loop:

while_loop
____________________________________



.. py:function:: paddle.static.nn.while_loop(cond, body, loop_vars, is_test=False, name=None)

该 API 用于实现类似 while 的循环控制功能，只要循环条件 ``cond`` 的返回值为 True，``while_loop`` 则会循环执行循环体 ``body``，直到 ``cond`` 的返回值为 False。

.. note::
    ``body`` 中定义的局部变量无法使用 ``Executor`` 的 ``fetch_list`` 来获取的，变量需在 ``body`` 外定义并将其置于 ``loop_vars`` 中进行循环更新后才可通过 ``fetch_list`` 获取。

参数
:::::::::

    - **cond** (callable) - 返回 boolean 类型 Tensor 的可调用函数，用以判断循环是否继续执行。``cond`` 的参数和 ``loop_vars`` 相对应。
    - **body** (callable) - 循环执行的结构体。其返回一个包含 tensor 或 LoDTensorArray 的列表或元组，且这些 tensor 或 LoDTensorArray 的长度，结构，类型和 ``loop_vars`` 中的相同。且``body`` 的参数与 ``loop_vars`` 相对应。
    - **loop_vars** (list|tuple) - 包含 tensor 或 LoDTensorArray 的列表或是元组，将其传入至 ``cond`` 和 ``body`` 中，得到循环条件和输出值。
    - **is_test** (bool，可选) - 用于表明是否在测试阶段执行，默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
list|tuple，循环迭代之后 ``body`` 的返回值，和 ``loop_vars`` 具有相同的结构。


示例代码
:::::::::

COPY-FROM: paddle.static.nn.while_loop
