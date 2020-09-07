.. _cn_api_fluid_dygraph_jit_load:

load
-----------------

.. py:function:: paddle.fluid.dygraph.jit.load(model_path, config=None)

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
    - **config** (SaveLoadConfig, 可选) - 用于指定额外配置选项的 :ref:`cn_api_fluid_dygraph_jit_SaveLoadConfig` 对象。默认为 ``None``。

返回：TranslatedLayer - 一个能够执行存储模型的 ``Layer`` 对象。

**示例代码**

1. 载入由接口 :ref:`cn_api_fluid_dygraph_jit_save` 存储的模型进行预测推理及fine-tune训练。

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

        # 2. load model

        # load
        loaded_layer = paddle.jit.load(model_path)

        # inference
        loaded_layer.eval()
        x = paddle.randn([1, IMAGE_SIZE], 'float32')
        pred = loaded_layer(x)

        # fine-tune
        loaded_layer.train()
        adam = opt.Adam(learning_rate=0.001, parameters=loaded_layer.parameters())
        train(loaded_layer, loader, loss_fn, adam)



2. 载入由接口 :ref:`cn_api_fluid_io_save_inference_model` 存储的模型进行预测推理及fine-tune训练。

    .. code-block:: python

        import numpy as np
        import paddle
        import paddle.fluid as fluid
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

        image = fluid.data(name='image', shape=[None, 784], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        pred = fluid.layers.fc(input=image, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=pred, label=label)
        avg_loss = fluid.layers.mean(loss)

        optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        optimizer.minimize(avg_loss)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = paddle.io.DataLoader(dataset,
            feed_list=[image, label],
            places=place,
            batch_size=BATCH_SIZE, 
            shuffle=True,
            drop_last=True,
            num_workers=2)

        # 1. train and save inference model
        for data in loader():
            exe.run(
                fluid.default_main_program(),
                feed=data, 
                fetch_list=[avg_loss])

        model_path = "fc.example.model"
        fluid.io.save_inference_model(
            model_path, ["image"], [pred], exe)

        # 2. load model

        # enable dygraph mode
        paddle.disable_static(place)

        # load
        fc = paddle.jit.load(model_path)

        # inference
        fc.eval()
        x = paddle.randn([1, IMAGE_SIZE], 'float32')
        pred = fc(x)

        # fine-tune
        fc.train()
        loss_fn = nn.CrossEntropyLoss()
        adam = opt.Adam(learning_rate=0.001, parameters=fc.parameters())
        loader = paddle.io.DataLoader(dataset,
            places=place,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2)
        for epoch_id in range(EPOCH_NUM):
            for batch_id, (image, label) in enumerate(loader()):
                out = fc(image)
                loss = loss_fn(out, label)
                loss.backward()
                adam.step()
                adam.clear_grad()
                print("Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, np.mean(loss.numpy())))

