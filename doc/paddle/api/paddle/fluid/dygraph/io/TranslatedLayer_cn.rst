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
        import paddle
        import paddle.nn as nn
        import paddle.optimizer as opt

        BATCH_SIZE = 16
        BATCH_NUM = 4
        EPOCH_NUM = 4

        IMAGE_SIZE = 784
        CLASS_NUM = 10

        # define a random dataset
        class RandomDataset(paddle.io.Dataset):
            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __getitem__(self, idx):
                image = np.random.random([IMAGE_SIZE]).astype('float32')
                label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                return image, label

            def __len__(self):
                return self.num_samples

        class LinearNet(nn.Layer):
            def __init__(self):
                super(LinearNet, self).__init__()
                self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

            @paddle.jit.to_static
            def forward(self, x):
                return self._linear(x)

        def train(layer, loader, loss_fn, opt):
            for epoch_id in range(EPOCH_NUM):
                for batch_id, (image, label) in enumerate(loader()):
                    out = layer(image)
                    loss = loss_fn(out, label)
                    loss.backward()
                    opt.step()
                    opt.clear_grad()
                    print("Epoch {} batch {}: loss = {}".format(
                        epoch_id, batch_id, np.mean(loss.numpy())))

        # enable dygraph mode
        place = paddle.CPUPlace()
        paddle.disable_static(place) 

        # 1. train & save model.

        # create network
        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()
        adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = paddle.io.DataLoader(dataset,
            places=place,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2)

        # train
        train(layer, loader, loss_fn, adam)

        # save
        model_path = "linear.example.model"
        paddle.jit.save(layer, model_path)

        # 2. load model as TranslatedLayer

        # load
        translated_layer = paddle.jit.load(model_path)

        # inference
        translated_layer.eval()
        x = paddle.randn([1, IMAGE_SIZE], 'float32')
        pred = translated_layer(x)

        # fine-tune
        translated_layer.train()
        adam = opt.Adam(learning_rate=0.001, parameters=translated_layer.parameters())
        train(translated_layer, loader, loss_fn, adam)


.. py:method:: program(method_name='forward'):

获取TranslatedLayer中指定方法对应的Program。

参数：
    - **method_name** (string) - 要获取的Porgram对应的方法名。默认值为"forward"。

返回：Program

返回类型：Program

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

        # enable dygraph mode
        fluid.enable_dygraph() 

        # 1. train & save model.
        # create network
        net = LinearNet(784, 1)
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
        # create data loader
        train_loader = fluid.io.DataLoader.from_generator(capacity=5)
        train_loader.set_batch_generator(random_batch_reader())
        # train
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

        # 2. load model as TranslatedLayer
        translated_layer = fluid.dygraph.jit.load(model_path)
        program = translated_layer.program()
