.. _cn_api_fluid_layers_create_py_reader_by_data:

create_py_reader_by_data
-------------------------------

.. py:function:: paddle.fluid.layers.create_py_reader_by_data(capacity,feed_list,name=None,use_double_buffer=True)

创建一个 Python reader用于在python中提供数据,该函数将返回一个 ``reader`` 变量。

它的工作方式与 ``py_reader`` 非常相似，除了它的输入是一个 feed_list 而不是 ``shapes``、 ``dtypes`` 和 ``lod_level``

参数：
  - **capacity** (int) - 缓冲区容量由 :code:`py_reader` 维护
  - **feed_list** (list(Variable)) - 传输数据列表
  - **name** (basestring) - 前缀Python队列名称和 reader 名称。不定义时将自动生成名称。
  - **use_double_buffer** (bool) - 是否使用 double buffer

返回： Variable: 一种reader，我们可以从中获得输入数据。

**代码示例：**

 :code:`py_reader` 的基本用法如下所示：

.. code-block:: python
    
    import paddle
    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist

    def network(img, label):
        # 用户自定义网络。此处以一个简单的线性回归作为示例。
        predict = fluid.layers.fc(input=img, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=predict, label=label)
        return fluid.layers.mean(loss)

    MEMORY_OPT = False
    USE_CUDA = False
    
    image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                   feed_list=[image, label])
    reader.decorate_paddle_reader(
        paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5), buf_size=500))
    img, label = fluid.layers.read_file(reader)
    loss = network(img, label) # 一些网络定义

    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = True if MEMORY_OPT else False
    compiled_prog = compiler.CompiledProgram(
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











