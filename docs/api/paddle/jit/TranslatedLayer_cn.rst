.. _cn_api_fluid_dygraph_TranslatedLayer:

TranslatedLayer
-------------------------------

.. py:class:: paddle.jit.TranslatedLayer(programs, persistable_vars)

``TranslatedLayer`` 是一个命令式编程模式 :ref:`cn_api_fluid_dygraph_Layer` 的继承类，
通过 :ref:`cn_api_paddle_jit_load` 载入构建。能够像一般 ``Layer`` 一样在 train 或者 eval 模式下使用。

.. note::
  ``TranslatedLayer`` 对象不能够通过构造函数创建，仅能够通过 :ref:`cn_api_paddle_jit_load` 接口载入构建。

代码示例
::::::::::::
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


方法
::::::::::::
program(method_name='forward'):
'''''''''

获取 TranslatedLayer 中指定方法对应的 Program。

**参数**

    - **method_name** (string) - 要获取的 Porgram 对应的方法名。默认值为"forward"。

**返回**
Program

**代码示例**
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
        model_path = "linear.example.model"
        paddle.jit.save(layer, model_path)

        # load
        translated_layer = paddle.jit.load(model_path)

        # get program
        program = translated_layer.program()
        print(program)
