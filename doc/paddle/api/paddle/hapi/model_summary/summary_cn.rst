.. _cn_api_paddle_summary:

summary
-------------------------------

.. py:function:: paddle.summary(self, input_size=None, dtype=None)

 ``summary`` 函数能够打印网络的基础结构和参数信息。

参数：
  - **net** (Layer) - 网络实例，必须是 ``Layer`` 的子类。
  - **input_size** (tuple|InputSpec|list[tuple|InputSpec) - 输入张量的大小。如果网络只有一个输入，那么该值需要设定为tuple或InputSpec。如果模型有多个输入。那么该值需要设定为list[tuple|InputSpec]，包含每个输入的shape。
  - **dtypes** (str，可选) - 输入张量的数据类型，如果没有给定，默认使用 ``float32`` 类型。默认值：None。

返回：字典，包含了总的参数量和总的可训练的参数量。

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.nn as nn

    class LeNet(nn.Layer):
        def __init__(self, num_classes=10):
            super(LeNet, self).__init__()
            self.num_classes = num_classes
            self.features = nn.Sequential(
                nn.Conv2D(
                    1, 6, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2D(2, 2),
                nn.Conv2D(
                    6, 16, 5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2D(2, 2))

            if num_classes > 0:
                self.fc = nn.Sequential(
                    nn.Linear(400, 120),
                    nn.Linear(120, 84),
                    nn.Linear(
                        84, 10))

        def forward(self, inputs):
            x = self.features(inputs)

            if self.num_classes > 0:
                x = paddle.flatten(x, 1)
                x = self.fc(x)
            return x

    lenet = LeNet()

    params_info = paddle.summary(lenet, (1, 1, 28, 28))
    print(params_info)
    # {'total_params': 61610, 'trainable_params': 61610}
