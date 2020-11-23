.. _cn_api_paddle_flops:

flops
-------------------------------

.. py:function:: paddle.flops(self, input_size=None, dtype=None)

 ``flops`` 函数能够打印网络的基础结构和参数信息。

参数：
  - **net** (Layer||Program) - 网络实例，必须是 ``Layer`` 的子类或者静态图下的Program。
  - **input_size** (list) - 输入张量的大小。注意：仅支持batch_size=1。
  - **custom_ops** (dict，可选) - 字典，用于实现对自定义网络层的统计。字典的key为自定义网络
                        层的class，value为统计网络层flops的函数，函数实现方法见示例代码。
                        此参数仅在 'net' 为Layer时生效。默认值：None。
  - **custom_ops** (bool, 可选) - bool值，用于控制是否打印每个网络层的细节。
返回：整型，网络模型的计算量。

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
        # m为网络层实例，x为网络层的输入，y为网络层的输出
        def count_leaky_relu(m, x, y):
            x = x[0]
            nelements = x.numel()
            m.total_ops += int(nelements)

        flops = paddle.flops(lenet, [1, 1, 28, 28], custom_ops= {nn.LeakyReLU: count_leaky_relu},
                            print_detail=True)
        print(flops)
