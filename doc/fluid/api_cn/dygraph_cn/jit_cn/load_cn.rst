.. _cn_api_fluid_dygraph_jit_load:

load
-----------------

.. py:function:: paddle.fluid.dygraph.jit.load(model_path, configs=None)

:api_attr: 命令式编程模式（动态图)

将接口 :ref:`cn_api_fluid_dygraph_jit_save` 或者 :ref:`cn_api_fluid_io_save_inference_model` 存储的模型载入为 :ref:`cn_api_fluid_dygraph_TranslatedLayer` ，用于预测推理或者fine-tune训练。

.. note::
    由于一些历史原因，如果载入的模型是通过 :ref:`cn_api_fluid_io_save_inference_model` 存储的，
    在使用它进行fine-tune训练时会存在一些局限：
    1. 命令式编程模式不支持 ``LoDTensor`` ，所有原先输入变量或者参数依赖于LoD信息的模型暂时无法使用；
    2. 所有存储模型的feed变量都需要被传入 ``Translatedlayer`` 的forward方法；
    3. 原模型变量的 ``stop_gradient`` 信息已丢失且无法准确恢复；
    4. 原模型参数的 ``trainable`` 信息已丢失且无法准确恢复。

参数：
    - **model_path** (str) - 存储模型的目录。
    - **configs** (SaveLoadConfig, 可选) - 用于指定额外配置选项的 :ref:`cn_api_fluid_dygraph_jit_SaveLoadConfig` 对象。默认为 ``None``。

返回：TranslatedLayer - 一个能够执行存储模型的 ``Layer`` 对象。

**示例代码**

1. 载入由接口 :ref:`cn_api_fluid_dygraph_jit_save` 存储的模型进行预测推理及fine-tune训练。

    .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid
        from paddle.fluid.dygraph import Linear
        from paddle.fluid.dygraph import declarative
        BATCH_SIZE = 32
        BATCH_NUM = 20
        def random_batch_reader():
            def _get_random_images_and_labels(image_shape, label_shape):
                image = np.random.random(size=image_shape).astype('float32')
                label = np.random.random(size=label_shape).astype('int64')
                return image, label
            def __reader__():
                for _ in range(BATCH_NUM):
                    batch_image, batch_label = _get_random_images_and_labels(
                        [BATCH_SIZE, 784], [BATCH_SIZE, 1])
                    yield batch_image, batch_label
            return __reader__
        class LinearNet(fluid.dygraph.Layer):
            def __init__(self, in_size, out_size):
                super(LinearNet, self).__init__()
                self._linear = Linear(in_size, out_size)
            @declarative
            def forward(self, x):
                return self._linear(x)
        # 开启命令式编程模式
        fluid.enable_dygraph() 
        # 1. 训练存储模型.
        # 创建网络
        net = LinearNet(784, 1)
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
        # 创建DataLoader
        train_loader = fluid.io.DataLoader.from_generator(capacity=5)
        train_loader.set_batch_generator(random_batch_reader())
        # 训练
        for data in train_loader():
            img, label = data
            label.stop_gradient = True
            cost = net(img)
            loss = fluid.layers.cross_entropy(cost, label)
            avg_loss = fluid.layers.mean(loss)
            avg_loss.backward()
            adam.minimize(avg_loss)
            net.clear_gradients()
        model_path = "linear.example.model"
        fluid.dygraph.jit.save(
            layer=net,
            model_path=model_path,
            input_spec=[img])
        # 2. 载入模型 & 预测
        # 载入模型
        infer_net = fluid.dygraph.jit.load(model_path)
        # 预测
        x = fluid.dygraph.to_variable(np.random.random((1, 784)).astype('float32'))
        pred = infer_net(x)
        # 3. 载入模型 & fine-tune训练
        # 载入模型
        train_net = fluid.dygraph.jit.load(model_path)
        train_net.train()
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=train_net.parameters())
        # 创建DataLoader
        train_loader = fluid.io.DataLoader.from_generator(capacity=5)
        train_loader.set_batch_generator(random_batch_reader())
        # fine-tune训练
        for data in train_loader():
            img, label = data
            label.stop_gradient = True
            cost = train_net(img)
            loss = fluid.layers.cross_entropy(cost, label)
            avg_loss = fluid.layers.mean(loss)
            avg_loss.backward()
            adam.minimize(avg_loss)
            train_net.clear_gradients()


2. 载入由接口 :ref:`cn_api_fluid_io_save_inference_model` 存储的模型进行预测推理及fine-tune训练。

    .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid
        BATCH_SIZE = 32
        BATCH_NUM = 20
        def random_batch_reader():
            def _get_random_images_and_labels(image_shape, label_shape):
                image = np.random.random(size=image_shape).astype('float32')
                label = np.random.random(size=label_shape).astype('int64')
                return image, label
            def __reader__():
                for _ in range(BATCH_NUM):
                    batch_image, batch_label = _get_random_images_and_labels(
                        [BATCH_SIZE, 784], [BATCH_SIZE, 1])
                    yield batch_image, batch_label
            return __reader__
        img = fluid.data(name='img', shape=[None, 784], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        pred = fluid.layers.fc(input=img, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=pred, label=label)
        avg_loss = fluid.layers.mean(loss)
        optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        optimizer.minimize(avg_loss)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[img, label], capacity=5, iterable=True)
        loader.set_batch_generator(random_batch_reader(), places=place)
        # 1. 训练 & 存储预测模型
        for data in loader():
            exe.run(
                fluid.default_main_program(),
                feed=data, 
                fetch_list=[avg_loss])
        model_path = "fc.example.model"
        fluid.io.save_inference_model(
            model_path, ["img"], [pred], exe)
        # 开启命令式编程模式
        fluid.enable_dygraph() 
        # 2. 载入模型 & 预测
        fc = fluid.dygraph.jit.load(model_path)
        x = fluid.dygraph.to_variable(np.random.random((1, 784)).astype('float32'))
        pred = fc(x)
        # 3. 载入模型 & fine-tune训练
        fc = fluid.dygraph.jit.load(model_path)
        fc.train()
        sgd = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=fc.parameters())
        train_loader = fluid.io.DataLoader.from_generator(capacity=5)
        train_loader.set_batch_generator(
            random_batch_reader(), places=place)
        for data in train_loader():
            img, label = data
            label.stop_gradient = True
            cost = fc(img)
            loss = fluid.layers.cross_entropy(cost, label)
            avg_loss = fluid.layers.mean(loss)
            avg_loss.backward()
            sgd.minimize(avg_loss)
