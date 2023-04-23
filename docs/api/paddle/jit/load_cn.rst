.. _cn_api_paddle_jit_load:

load
-----------------

.. py:function:: paddle.jit.load(path, **configs)


将接口 ``paddle.jit.save`` 或者 ``paddle.static.save_inference_model`` 存储的模型载入为 ``paddle.jit.TranslatedLayer``，用于预测推理或者 fine-tune 训练。

.. note::
    如果载入的模型是通过 ``paddle.static.save_inference_model`` 存储的，在使用它进行 fine-tune 训练时会存在一些局限：
    1. 命令式编程模式不支持 ``LoDTensor``，所有原先输入变量或者参数依赖于 LoD 信息的模型暂时无法使用；
    2. 所有存储模型的 feed 变量都需要被传入 ``Translatedlayer`` 的 forward 方法；
    3. 原模型变量的 ``stop_gradient`` 信息已丢失且无法准确恢复；
    4. 原模型参数的 ``trainable`` 信息已丢失且无法准确恢复。

参数
:::::::::
    - **path** (str) - 载入模型的路径前缀。格式为 ``dirname/file_prefix`` 或者 ``file_prefix`` 。
    - **config** (dict，可选) - 其他用于兼容的载入配置选项。这些选项将来可能被移除，如果不是必须使用，不推荐使用这些配置选项。默认为 ``None``。目前支持以下配置选项：
        (1) model_filename (str) - paddle 1.x 版本 ``save_inference_model`` 接口存储格式的预测模型文件名，原默认文件名为 ``__model__`` ；
        (2) params_filename (str) - paddle 1.x 版本 ``save_inference_model`` 接口存储格式的参数文件名，没有默认文件名，默认将各个参数分散存储为单独的文件。

返回
:::::::::
TranslatedLayer，一个能够执行存储模型的 ``Layer`` 对象。

代码示例
:::::::::

1. 载入由接口 ``paddle.jit.save`` 存储的模型进行预测推理及 fine-tune 训练。

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
                super().__init__()
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

        # 1. train & save model.

        # create network
        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()
        adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = paddle.io.DataLoader(dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2)

        # train
        train(layer, loader, loss_fn, adam)

        # save
        path = "example_model/linear"
        paddle.jit.save(layer, path)

        # 2. load model

        # load
        loaded_layer = paddle.jit.load(path)

        # inference
        loaded_layer.eval()
        x = paddle.randn([1, IMAGE_SIZE], 'float32')
        pred = loaded_layer(x)

        # fine-tune
        loaded_layer.train()
        adam = opt.Adam(learning_rate=0.001, parameters=loaded_layer.parameters())
        train(loaded_layer, loader, loss_fn, adam)



2. 兼容载入由接口 ``paddle.fluid.io.save_inference_model`` 存储的模型进行预测推理及 fine-tune 训练。

    .. code-block:: python

        import numpy as np
        import paddle
        import paddle.static as static
        import paddle.nn as nn
        import paddle.optimizer as opt
        import paddle.nn.functional as F

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

        paddle.enable_static()

        image = static.data(name='image', shape=[None, 784], dtype='float32')
        label = static.data(name='label', shape=[None, 1], dtype='int64')
        pred = static.nn.fc(x=image, size=10, activation='softmax')
        loss = F.cross_entropy(input=pred, label=label)
        avg_loss = paddle.mean(loss)

        optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        optimizer.minimize(avg_loss)

        place = paddle.CPUPlace()
        exe = static.Executor(place)
        exe.run(static.default_startup_program())

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = paddle.io.DataLoader(dataset,
            feed_list=[image, label],
            places=place,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            return_list=False,
            num_workers=2)

        # 1. train and save inference model
        for data in loader():
            exe.run(
                static.default_main_program(),
                feed=data,
                fetch_list=[avg_loss])

        model_path = "fc.example.model"
        paddle.fluid.io.save_inference_model(
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
