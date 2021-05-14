.. _cn_api_fluid_io_load_from_file:

load_from_file
-------------------------------


.. py:function:: paddle.static.load_from_file(path)



从指定的文件中加载内容。

参数：
  - **path** (str) - 要读取的文件。

返回:
  - bytes: 文件内容。

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

    # 将序列化之后的参数保存到文件
    params_path = path_prefix + ".params"
    paddle.static.save_to_file(params_path, serialized_params)

    # 从文件加载序列化之后的参数
    serialized_params_copy = paddle.static.load_from_file(params_path)
