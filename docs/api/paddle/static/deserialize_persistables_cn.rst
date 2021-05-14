.. _cn_api_fluid_io_deserialize_persistables:

deserialize_persistables
-------------------------------


.. py:function:: paddle.static.deserialize_persistables(program, data, executor)




根据指定的 program 和 executor，反序列化模型参数。

参数：
  - **program** (Program) - 指定包含要反序列化的参数的名称的 program。
  - **data** (bytes) - 序列化之后的模型参数。
  - **executor** (Executor) - 用来执行 load op 的 ``executor`` 。 

返回:
  - Program: 包含反序列化后的参数的program。

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

    # 序列化参数
    serialized_params = paddle.static.serialize_persistables([image], [predict], exe)

    # 反序列化成参数
    main_program = paddle.static.default_main_program()
    deserialized_params = paddle.static.deserialize_persistables(main_program, serialized_params, exe)
