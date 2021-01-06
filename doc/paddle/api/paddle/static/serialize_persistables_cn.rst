.. _cn_api_fluid_io_serialize_persistables:

serialize_persistables
-------------------------------


.. py:function:: paddle.static.serialize_persistables(feed_vars, fetch_vars, executor, **kwargs)




根据指定的 feed_vars，fetch_vars 和 executor，序列化模型参数。

参数：
  - **feed_vars** (Variable | list[Variable]) – 模型的输入变量。
  - **fetch_vars** (Variable | list[Variable]) – 模型的输出变量。
  - **executor** (Executor) - 用于保存预测模型的 ``executor`` ，详见 :ref:`api_guide_executor` 。 
  - **kwargs** - 支持的 key 包括 'program'。(注意：kwargs 主要是用来做反向兼容的)
      - **program** - 指定包含要序列化的参数的 program，默认是 default_main_program。

返回：参数序列化之后的字节数组。

抛出异常：
  - ``ValueError`` – 如果 ``feed_vars`` 或 ``fetch_vars`` 类型不是 Variable 或 list[Variable]，则抛出异常。

**代码示例**

.. code-block:: python

    import paddle

    paddle.enable_static()

    path_prefix = "./infer_model"

    # 用户自定义网络, 此处用 softmax 回归为例
    image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    predict = paddle.static.nn.fc(image, 10, activation='softmax')

    loss = paddle.nn.functional.cross_entropy(predict, label)

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(paddle.static.default_startup_program())

    # 将参数序列化成字节数组
    serialized_params = paddle.static.serialize_persistables([image], [predict], exe)
