.. _cn_api_fluid_layers_create_py_reader_by_data:

create_py_reader_by_data
-------------------------------


.. py:function:: paddle.fluid.layers.create_py_reader_by_data(capacity,feed_list,name=None,use_double_buffer=True)

:api_attr: 声明式编程模式（静态图)



创建一个Python端提供数据的reader。该OP与 :ref:`cn_api_fluid_layers_py_reader` 类似，不同点在于它能够从feed变量列表读取数据。

参数：
  - **capacity** (int) - ``py_reader`` 维护的队列缓冲区的容量大小。单位是batch数量。若reader读取速度较快，建议设置较大的 ``capacity`` 值。
  - **feed_list** (list(Variable)) - feed变量列表，这些变量一般由 :code:`fluid.data()` 创建。
  - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
  - **use_double_buffer** (bool，可选) - 是否使用双缓冲区，双缓冲区是为了预读下一个batch的数据、异步CPU -> GPU拷贝。默认值为True。

返回：能够从feed变量列表读取数据的reader，数据类型和feed变量列表中变量的数据类型相同。

返回类型：reader

**代码示例：**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist

    def network(img, label):
        # 用户构建自定义网络，此处以一个简单的线性回归为例。
        predict = fluid.layers.fc(input=img, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=predict, label=label)
        return fluid.layers.mean(loss)

    MEMORY_OPT = False
    USE_CUDA = False

    image = fluid.data(name='image', shape=[None, 1, 28, 28], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                   feed_list=[image, label])
    reader.decorate_paddle_reader(
        paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5), buf_size=500))
    img, label = fluid.layers.read_file(reader)
    loss = network(img, label) # 用户构建自定义网络并返回损失函数

    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = True if MEMORY_OPT else False
    exec_strategy = fluid.ExecutionStrategy()
    compiled_prog = fluid.compiler.CompiledProgram(
        fluid.default_main_program()).with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    for epoch_id in range(2):
        reader.start()
        try:
            while True:
                exe.run(compiled_prog, fetch_list=[loss.name])
        except fluid.core.EOFException:
            reader.reset()
