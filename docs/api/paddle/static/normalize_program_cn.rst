.. _cn_api_fluid_io_normalize_program:

normalize_program
-------------------------------


.. py:function:: paddle.static.normalize_program(program, feed_vars, fetch_vars)




根据指定的 feed_vars 和 fetch_vars，优化 program。

参数：
  - **program** - 指定想要优化的 program。
  - **feed_vars** (Variable | list[Variable]) – 模型的输入变量。
  - **fetch_vars** (Variable | list[Variable]) – 模型的输出变量。

返回：优化之后的 program。

抛出异常：
  - ``TypeError`` – 如果 ``program`` 类型不是 ``Program``, 或 ``feed_vars``, ``fetch_vars`` 类型不是 Variable 或 list[Variable]，则抛出异常。

**代码示例**

.. code-block:: python

    import paddle

    paddle.enable_static()

    path_prefix = "./infer_model"

    # User defined network, here a softmax regession example
    image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    predict = paddle.static.nn.fc(image, 10, activation='softmax')

    loss = paddle.nn.functional.cross_entropy(predict, label)

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(paddle.static.default_startup_program())

    # normalize main program.
    program = paddle.static.default_main_program()
    normalized_program = paddle.static.normalize_program(program, [image], [predict])

