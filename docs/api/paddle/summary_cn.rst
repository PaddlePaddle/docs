.. _cn_api_paddle_summary:

summary
-------------------------------

.. py:function:: paddle.summary(net, input_size, dtypes=None)

 ``summary`` 函数能够打印网络的基础结构和参数信息。

参数：
  - **net** (Layer) - 网络实例，必须是 ``Layer`` 的子类。
  - **input_size** (tuple|InputSpec|list[tuple|InputSpec) - 输入张量的大小。如果网络只有一个输入，那么该值需要设定为tuple或InputSpec。如果模型有多个输入。那么该值需要设定为list[tuple|InputSpec]，包含每个输入的shape。
  - **dtypes** (str，可选) - 输入张量的数据类型，如果没有给定，默认使用 ``float32`` 类型。默认值：None。
  - **input** (tensor，可选) - 输入张量数据，如果给出``input``，那么``input_size``和``input_size``的输入将被忽略。默认值：None。

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
    # ---------------------------------------------------------------------------
    # Layer (type)       Input Shape          Output Shape         Param #    
    # ===========================================================================
    # Conv2D-11      [[1, 1, 28, 28]]      [1, 6, 28, 28]          60       
    #     ReLU-11       [[1, 6, 28, 28]]      [1, 6, 28, 28]           0       
    # MaxPool2D-11     [[1, 6, 28, 28]]      [1, 6, 14, 14]           0       
    # Conv2D-12      [[1, 6, 14, 14]]     [1, 16, 10, 10]         2,416     
    #     ReLU-12      [[1, 16, 10, 10]]     [1, 16, 10, 10]           0       
    # MaxPool2D-12    [[1, 16, 10, 10]]      [1, 16, 5, 5]            0       
    # Linear-16         [[1, 400]]            [1, 120]           48,120     
    # Linear-17         [[1, 120]]            [1, 84]            10,164     
    # Linear-18         [[1, 84]]             [1, 10]              850      
    # ===========================================================================
    # Total params: 61,610
    # Trainable params: 61,610
    # Non-trainable params: 0
    # ---------------------------------------------------------------------------
    # Input size (MB): 0.00
    # Forward/backward pass size (MB): 0.11
    # Params size (MB): 0.24
    # Estimated Total Size (MB): 0.35
    # ---------------------------------------------------------------------------
    # {'total_params': 61610, 'trainable_params': 61610}

    # multi input demo
    class LeNetMultiInput(LeNet):
        def forward(self, inputs, y):
            x = self.features(inputs)
            if self.num_classes > 0:
                x = paddle.flatten(x, 1)
                x = self.fc(x + y)
            return x
    
    lenet_multi_input = LeNetMultiInput()
    params_info = paddle.summary(lenet_multi_input, [(1, 1, 28, 28), (1, 400)], 
                                ['float32', 'float32'])
    print(params_info)

    # list input demo
    class LeNetListInput(LeNet):

        def forward(self, inputs):
            x = self.features(inputs[0])

            if self.num_classes > 0:
                x = paddle.flatten(x, 1)
                x = self.fc(x + inputs[1])
            return x

    lenet_list_input = LeNetListInput()
    input_data = [paddle.rand([1, 1, 28, 28]), paddle.rand([1, 400])]
    params_info = paddle.summary(lenet_list_input, input=input_data)
    print(params_info)

    # dict input demo
    class LeNetDictInput(LeNet):

        def forward(self, inputs):
            x = self.features(inputs['x1'])

            if self.num_classes > 0:
                x = paddle.flatten(x, 1)
                x = self.fc(x + inputs['x2'])
            return x

    lenet_dict_input = LeNetDictInput()
    input_data = {'x1': paddle.rand([1, 1, 28, 28]),
              'x2': paddle.rand([1, 400])}
    params_info = paddle.summary(lenet_dict_input, input=input_data)
    print(params_info)

