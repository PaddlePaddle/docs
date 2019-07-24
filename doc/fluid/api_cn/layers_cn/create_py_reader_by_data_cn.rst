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
    
    image = fluid.layers.data(name='image', shape=[1, 28, 28], dtypes='float32')
    label = fluid.layers.data(name='label', shape=[1], dtypes='int64')
    reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                   feed_list=[image, label])
    reader.decorate_paddle_reader(
        paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5), buf_size=500))
    img, label = fluid.layers.read_file(reader)
    loss = network(img, label) # 一些网络定义

    fluid.Executor(fluid.CUDAPlace(0)).run(fluid.default_startup_program())

    exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)
    for epoch_id in range(10):
        reader.start()
        try:
            while True:
                exe.run(fetch_list=[loss.name])
        except fluid.core.EOFException:
            reader.reset()











