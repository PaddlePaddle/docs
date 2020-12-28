.. _cn_api_fluid_io_serialize_program:

serialize_program
-------------------------------


.. py:function:: paddle.static.serialize_program(feed_vars, fetch_vars, **kwargs)




根据指定的 feed_vars 和 fetch_vars，序列化 program。

参数：
  - **feed_vars** (Variable | list[Variable]) – 模型的输入变量。
  - **fetch_vars** (Variable | list[Variable]) – 模型的输出变量。
  - **kwargs** - 支持的 key 包括 'program'。(注意：kwargs 主要是用来做反向兼容的)
    - **program** - 指定想要序列化的 program，默认是 default_main_program。

返回：字节数组。

抛出异常：
  - ``ValueError`` – 如果 ``feed_vars`` 或 ``fetch_vars`` 类型不是 Variable 或 list[Variable]，则抛出异常。

**代码示例**

.. code-block:: python

    import paddle

    paddle.enable_static()

    path_prefix = "./infer_model"

    # 用户自定义网络，此处用 softmax 回归为例。
    image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    predict = paddle.static.nn.fc(image, 10, activation='softmax')

    loss = paddle.nn.functional.cross_entropy(predict, label)

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(paddle.static.default_startup_program())

    # 序列化program
    serialized_program = paddle.static.serialize_program([image], [predict])

    # 反序列化program
    deserialized_program = paddle.static.deserialize_program(serialized_program)
