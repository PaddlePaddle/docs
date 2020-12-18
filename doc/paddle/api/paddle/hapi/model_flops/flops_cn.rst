.. _cn_api_paddle_flops:

flops
-------------------------------

.. py:function:: paddle.flops(self, input_size=None, dtype=None)

 ``flops`` 函数能够打印网络的基础结构和参数信息。

参数：
  - **net** (paddle.nn.Layer||paddle.static.Program) - 网络实例，必须是 paddle.nn.Layer
                        的子类或者静态图下的 paddle.static.Program。
  - **input_size** (list) - 输入张量的大小。注意：仅支持batch_size=1。
  - **custom_ops** (dict，可选) - 字典，用于实现对自定义网络层的统计。字典的key为自定义网络
                        层的class，value为统计网络层flops的函数，函数实现方法见示例代码。
                        此参数仅在 'net' 为paddle.nn.Layer时生效。默认值：None。
  - **print_detail (bool, 可选) - bool值，用于控制是否打印每个网络层的细节。默认值：False
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
        # m 是 nn.Layer 的一个实类, x 是m的输入, y 是网络层的输出.
        def count_leaky_relu(m, x, y):
            x = x[0]
            nelements = x.numel()
            m.total_ops += int(nelements)

        FLOPs = paddle.flops(lenet, [1, 1, 28, 28], custom_ops= {nn.LeakyReLU: count_leaky_relu},
                            print_detail=True)
        print(FLOPs)

        #+--------------+-----------------+-----------------+--------+--------+
        #|  Layer Name  |   Input Shape   |   Output Shape  | Params | Flops  |
        #+--------------+-----------------+-----------------+--------+--------+
        #|   conv2d_2   |  [1, 1, 28, 28] |  [1, 6, 28, 28] |   60   | 47040  |
        #|   re_lu_2    |  [1, 6, 28, 28] |  [1, 6, 28, 28] |   0    |   0    |
        #| max_pool2d_2 |  [1, 6, 28, 28] |  [1, 6, 14, 14] |   0    |   0    |
        #|   conv2d_3   |  [1, 6, 14, 14] | [1, 16, 10, 10] |  2416  | 241600 |
        #|   re_lu_3    | [1, 16, 10, 10] | [1, 16, 10, 10] |   0    |   0    |
        #| max_pool2d_3 | [1, 16, 10, 10] |  [1, 16, 5, 5]  |   0    |   0    |
        #|   linear_0   |     [1, 400]    |     [1, 120]    | 48120  | 48000  |
        #|   linear_1   |     [1, 120]    |     [1, 84]     | 10164  | 10080  |
        #|   linear_2   |     [1, 84]     |     [1, 10]     |  850   |  840   |
        #+--------------+-----------------+-----------------+--------+--------+
        #Total Flops: 347560     Total Params: 61610
