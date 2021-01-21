.. _cn_api_paddle_jit_save:

save
-----------------

.. py:function:: paddle.jit.save(layer, path, input_spec=None, **configs)

将输入的 ``Layer`` 存储为 ``paddle.jit.TranslatedLayer`` 格式的模型，载入后可用于预测推理或者fine-tune训练。

该接口会将输入 ``Layer`` 转写后的模型结构 ``Program`` 和所有必要的持久参数变量存储至输入路径 ``path`` 。

``path`` 是存储目标的前缀，存储的模型结构 ``Program`` 文件的后缀为 ``.pdmodel`` ，存储的持久参数变量文件的后缀为 ``.pdiparams`` ，同时这里也会将一些变量描述信息存储至文件，文件后缀为 ``.pdiparams.info`` ，这些额外的信息将在fine-tune训练中使用。

存储的模型能够被以下API完整地载入使用：
    - ``paddle.jit.load``
    - ``paddle.static.load_inference_model`` 
    - 其他预测库API

参数
:::::::::
    - layer (Layer) - 需要存储的 ``Layer`` 对象。
    - path (str) - 存储模型的路径前缀。格式为 ``dirname/file_prefix`` 或者 ``file_prefix`` 。
    - input_spec (list[InputSpec|Tensor], 可选) - 描述存储模型forward方法的输入，可以通过InputSpec或者示例Tensor进行描述。如果为 ``None`` ，所有原 ``Layer`` forward方法的输入变量将都会被配置为存储模型的输入变量。默认为 ``None``。
    - **configs (dict, 可选) - 其他用于兼容的存储配置选项。这些选项将来可能被移除，如果不是必须使用，不推荐使用这些配置选项。默认为 ``None``。目前支持以下配置选项：(1) output_spec (list[Tensor]) - 选择存储模型的输出目标。默认情况下，所有原 ``Layer`` forward方法的返回值均会作为存储模型的输出。如果传入的 ``output_spec`` 列表不是所有的输出变量，存储的模型将会根据 ``output_spec`` 所包含的结果被裁剪。

返回
:::::::::
无

代码示例
:::::::::

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
