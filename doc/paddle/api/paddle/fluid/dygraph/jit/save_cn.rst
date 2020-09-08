.. _cn_api_fluid_dygraph_jit_save:

save
-----------------

.. py:function:: paddle.jit.save(layer, model_path, input_spec=None, config=None)

将输入的经过 ``@declarative`` 装饰的 :ref:`cn_api_fluid_dygraph_Layer` 存储为 :ref:`cn_api_fluid_dygraph_TranslatedLayer` 格式的模型，
载入后可用于预测推理或者fine-tune训练。

该接口将会将输入 :ref:`cn_api_fluid_dygraph_Layer` 转写后的模型结构 ``Program`` 和所有必要的持久参数变量存储至输入路径 ``model_path`` 中。

默认存储的 ``Program`` 文件名为 ``__model__``， 默认存储持久参数变量的文件名为 ``__variables__``，
同时会将变量的一些描述信息存储至文件 ``__variables.info__``，这些额外的信息将在fine-tune训练中使用。

存储的模型能够被以下API载入使用：
  - :ref:`cn_api_fluid_dygraph_jit_load`
  - :ref:`cn_api_fluid_io_load_inference_model` （需要配置参数 ``params_filename='__variables__'`` ）
  - 其他预测库API

参数：
    - **layer** (Layer) - 需要存储的 :ref:`cn_api_fluid_dygraph_Layer` 对象。输入的 ``Layer`` 需要经过 ``@declarative`` 装饰。
    - **model_path** (str) - 存储模型的目录。
    - **input_spec** (list[Variable], 可选) - 描述存储模型的输入。此参数是传入当前存储的 ``TranslatedLayer`` forward方法的一个示例输入。如果为 ``None`` ，所有原 ``Layer`` forward方法的输入变量将都会被配置为存储模型的输入变量。默认为 ``None``。
    - **config** (SaveLoadConfig, 可选) - 用于指定额外配置选项的 :ref:`cn_api_fluid_dygraph_jit_SaveLoadConfig` 对象。默认为 ``None``。

返回：无

**示例代码**

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

