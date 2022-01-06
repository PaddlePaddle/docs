# Build the Model

After creating dataset, you need to build your model. First, this tutorial introduces the APIs for building model in Paddle.Then introduces two methods of building models supported by Paddle, namely Sequential and SubClass. Finally, introduce paddle provides pre-trained models.

# 1. paddle.nn

After Paddle 2.0, the building model APIs are included under the `paddle.nn`, and you can build specific models by Sequential or SubClass. The list of APIs are categorized in the following table.

| function | name |
| --- | --- | 
| Conv | Conv1D、Conv2D、Conv3D、Conv1DTranspose、Conv2DTranspose、Conv3DTranspose |
| Pool | AdaptiveAvgPool1D、AdaptiveAvgPool2D、AdaptiveAvgPool3D、 AdaptiveMaxPool1D、AdaptiveMaxPool2D、AdaptiveMaxPool3D、 AvgPool1D、AvgPool2D、AvgPool3D、MaxPool1D、MaxPool2D、MaxPool3D |
| Padding | Pad1D、Pad2D、Pad3D |
| Activation | ELU、GELU、Hardshrink、Hardtanh、HSigmoid、LeakyReLU、LogSigmoid、 LogSoftmax、PReLU、ReLU、ReLU6、SELU、Sigmoid、Softmax、Softplus、 Softshrink、Softsign、Tanh、Tanhshrink |
| Normlization |BatchNorm、BatchNorm1D、BatchNorm2D、BatchNorm3D、GroupNorm、 InstanceNorm1D、InstanceNorm2D、InstanceNorm3D、LayerNorm、SpectralNorm、 SyncBatchNorm |
| Recurrent NN | BiRNN、GRU、GRUCell、LSTM、LSTMCell、RNN、RNNCellBase、SimpleRNN、 SimpleRNNCell |
| Transformer | Transformer、TransformerDecoder、TransformerDecoderLayer、 TransformerEncoder、TransformerEncoderLayer |
| Dropout | AlphaDropout、Dropout、Dropout2D、Dropout3D |
| Loss | BCELoss、BCEWithLogitsLoss、CrossEntropyLoss、CTCLoss、KLDivLoss、L1Loss MarginRankingLoss、MSELoss、NLLLoss、SmoothL1Loss |

## 2. Build The Model with Sequential

For linear network structure, you can directly use `Sequential` to quickly build the model, reducing the definition of classes and other code. The code is as follows:


```python
import paddle
# Build The Model with Sequential
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)
```

## 3. Build The Model with SubClass

Generally, you can build the model with SubClass，declare the Model's Layer in `__init__` and define the forward calculation in `forward`.


```python
## Build The Model with SubClass

class Mnist(paddle.nn.Layer):
    def __init__(self):
        super(Mnist, self).__init__()

        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(784, 512)
        self.linear_2 = paddle.nn.Linear(512, 10)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)

    def forward(self, inputs):
        y = self.flatten(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)

        return y

mnist_2 = Mnist()
```

## 4. Paddle provides pre-trained Model

In addition to building the models using the above method, you can also directly use the models provided by Paddle in `paddle.vision.models`, as listed below.


```python
print('models provided of Paddle: ', paddle.vision.models.__all__)
```

    models provided of Paddle:  ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV1', 'mobilenet_v1', 'MobileNetV2', 'mobilenet_v2', 'LeNet']


Use as follows:


```python
lenet = paddle.vision.models.LeNet()
```

You can view the structure of the model and output shapes by using the `paddle.summary()`.


```python

```
