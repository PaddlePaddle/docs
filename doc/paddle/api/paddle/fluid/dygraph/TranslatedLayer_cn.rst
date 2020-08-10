.. _cn_api_fluid_dygraph_TranslatedLayer:

TranslatedLayer
-------------------------------

.. py:class:: paddle.fluid.dygraph.TranslatedLayer(programs, persistable_vars)

``TranslatedLayer`` 是一个命令式编程模式 :ref:`cn_api_fluid_dygraph_Layer` 的继承类，
通过 :ref:`cn_api_fluid_dygraph_jit_load` 载入构建。能够像一般 ``Layer`` 一样在train或者eval模式下使用。

.. note::
  ``TranslatedLayer`` 对象不能够通过构造函数创建，仅能够通过 :ref:`cn_api_fluid_dygraph_jit_load` 接口载入构建。

**示例代码：**
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
        # 2. 载入模型构建TranslatedLayer
        translated_layer = fluid.dygraph.jit.load(model_path)
        # 预测
        translated_layer.eval()
        x = fluid.dygraph.to_variable(np.random.random((1, 784)).astype('float32'))
        pred = translated_layer(x)
        # fine-tune训练
        translated_layer.train()
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=translated_layer.parameters())
        train_loader = fluid.io.DataLoader.from_generator(capacity=5)
        train_loader.set_batch_generator(random_batch_reader())
        for data in train_loader():
            img, label = data
            label.stop_gradient = True
            cost = translated_layer(img)
            loss = fluid.layers.cross_entropy(cost, label)
            avg_loss = fluid.layers.mean(loss)
            avg_loss.backward()
            adam.minimize(avg_loss)
            translated_layer.clear_gradients()
