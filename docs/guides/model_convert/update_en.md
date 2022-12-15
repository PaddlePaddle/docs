# Upgrade guide

## Upgrade summary
PaddlePaddle 2.0 has significant upgrades, involving important changes in development as follows.

 - Dynamic graph function is improved, the concept of data representation under the dynamic graph mode is `Tensor`. It is recommended to use the dynamic graph mode;
 - API directory system is adjusted. API naming and aliases have been unified and standardized. Although the old API is compatible, please use the new API system for development;
 - The data processing, networking method, model training, multi-card startup, model saving and inference have been optimized correspondingly, please check the description correspondingly;

Please read this guide carefully for the above changes. For the upgrade of existing models, PaddlePaddle also provides conversion tool 2.0  (see Appendix).
Some other feature additions such as dynamic graphs for quantization training, blending accuracy support, and dynamic-static conversion are not listed here, but can be found in the [Release Note](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/ release_note_cn.html) or the corresponding documentation.

## I. Dynamic graph

### Recommend to use dynamic graph mode in preference
PaddlePaddle 2.0 will use dynamic graph as the default mode (if you still want to use static graph, you can switch it by calling ``paddle.enable_static``).。

```python
import paddle
```

### Use Tensor concept to represent data
In static graph mode, ``Variable`` is used to represent data because the data used in networking is not accessible in real time.
In dynamic graph mode, the data representation concept is unified as `Tensor` for intuitive reasons, etc. There are two main ways to create `Tensor`:

1. Convert `python scalar/list`, or `numpy.ndarray` data to Paddle `Tensor` by `paddle.to_tensor` function. Please check the API on the official website for details.

```python
import paddle
import numpy as np

paddle.to_tensor(1)
paddle.to_tensor((1.1, 2.2))
paddle.to_tensor(np.random.randn(3, 4))
```

2. Create and return back to Tensor by  `paddle.zeros, paddle.ones, paddle.full, paddle.arange, paddle.rand, paddle.randn, paddle.randint, paddle.normal, paddle.uniform` functions.

## II. API
### API directory structure

In order to make the API organization more concise and clear, the original directory system of paddle.fluid.xxx is upgraded to paddle.xxx, and the organization of subdirectories is systematically organized and optimized. It also adds high-level APIs. paddle.fluid directory temporarily retains version 1.8 APIs, mainly for compatibility reasons, which will be removed in the future.

**For 2.0 development tasks, please use the APIs in the paddle directory rather than the APIs in the paddle.fluid directory.** If you find any APIs missing in the Paddle directory, it is recommended to use the base APIs for a combined implementation; you can also use the APIs in [github](https://github.com /paddlepaddle/paddle) to give feedback.

**The overall directory structure of API 2.0 is as follows:**

| Directory | Function and API |
| :--- | --------------- |
| paddle.*          | The paddle root directory retains aliases for common APIs, currently including: paddle.tensor, paddle.framework, and all APIs in the paddle.device directory |
| paddle.tensor     | APIs related to tensor operations, such as creating zeros, matmul, concat, adding, finding argmax, etc. |
| paddle.framework  | Framework general API and dynamic graph model API, such as no_grad, save, load, etc. |
| paddle.device     | Device management APIs, such as set_device, get_device, etc. |
| paddle.amp        | paddle auto-mixing accuracy strategies, including auto_cast, GradScaler, etc. |
| paddle.callbacks  | paddle logging callback classes, including ModelCheckpoint, ProgBarLogger, etc. |
| paddle.nn         | Network-related APIs, such as Linear, Convolutional Conv2D, Recurrent Neural Network LSTM, Loss Function CrossEntropyLoss, Activation Function ReLU, etc.Network-related APIs, such as Linear, Convolutional Conv2D, Recurrent Neural Network LSTM, Loss Function CrossEntropyLoss, Activation Function ReLU, etc. |
| paddle.static     | APIs related to the underlying framework under static graphs, such as Variable, Program, Executor, etc. |
| paddle.static.nn  | APIs for networking under static graphs, such as full connection layer fc, control flow while_loop/cond. |
| paddle.optimizer  | Optimization algorithm related APIs, e.g. SGD, Adagrad, Adam, etc. |
| paddle.optimizer.lr  | APIs related to learning rate decay, such as NoamDecay, StepDecay, PiecewiseDecay, etc. |
| paddle.metric     | APIs related to evaluation metrics calculation, e.g. Accuracy, Auc, etc. |
| paddle.io         | Data input and output related APIs, such as Dataset, DataLoader, etc. |
| paddle.distributed      | Distributed base APIs                           |
| paddle.distributed.fleet      | Distributed high-level APIs          |
| paddle.vision     | Vision domain APIs, such as dataset Cifar10, data processing ColorJitter, common underlying network structure ResNet, etc. |
| paddle.text       | Currently includes NLP domain related datasets, such as Imdb, Movielens. |

### API alias rule

- APIs are created with aliases in different paths for better convinience:
    - All APIs under device, framework, and tensor directories are aliased in the paddle root directory; all APIs are not aliased in the paddle root directory except a few special APIs.
    - All APIs in the paddle.nn directory except for the functional directory have aliases in the paddle.nn directory; all APIs in the functional directory have no aliases in the paddle.nn directory.
- ** **It is recommended to give preference to aliases with shorter paths**, for example `paddle.add -> paddle.tensor.add`;  `paddle.add` is recommended.
- The following are special alias relationships and the left API name are recommended:
  - paddle.tanh -> paddle.tensor.tanh -> paddle.nn.functional.tanh
  - paddle.remainder -> paddle.mod -> paddle.floor_mod
  - paddle.rand -> paddle.uniform
  - paddle.randn -> paddle.standard_normal
  - Layer.set_state_dict -> Layer.set_dict

### Common API name changes

- Use full names for addition, subtraction, multiplication and division, not abbreviations
- For current element-wise operation, do not add elementwise prefix
- For operation by an axis, do not add reduce prefix
- Conv, Pool, Dropout, BatchNorm, Pad networking APIs add 1D, 2D, 3D suffixes according to input data type

  | Paddle 1.8  API Names | Paddle 2.0 对应的名称|
  | --------------- | ------------------------ |
  | paddle.fluid.layers.elementwise_add | paddle.add               |
  | paddle.fluid.layers.elementwise_sub | paddle.subtract          |
  | paddle.fluid.layers.elementwise_mul | paddle.multiply          |
  | paddle.fluid.layers.elementwise_div | paddle.divide |
  | paddle.fluid.layers.elementwise_max | paddle.maximum             |
  | paddle.fluid.layers.elementwise_min | paddle.minimum |
  | paddle.fluid.layers.reduce_sum | paddle.sum |
  | paddle.fluid.layers.reduce_prod | paddle.prod |
  | paddle.fluid.layers.reduce_max | paddle.max        |
  | paddle.fluid.layers.reduce_min | paddle.min        |
  | paddle.fluid.layers.reduce_all | paddle.all        |
  | paddle.fluid.layers.reduce_any | paddle.any        |
  | paddle.fluid.dygraph.Conv2D | paddle.nn.Conv2D |
  | paddle.fluid.dygraph.Conv2DTranspose | paddle.nn.Conv2DTranspose |
  | paddle.fluid.dygraph.Pool2D | paddle.nn.MaxPool2D, paddle.nn.AvgPool2D |

## III. Development process
### Data processing
**Dataset, Sampler, BatchSampler, DataLoader interfaces** in the **paddle.io** directory are recommended for data processing, reader-like interfaces are not recommended. Some common datasets have been implemented in the paddle.vision.datasets and paddle.text.datasets directories, refer to the API documentation for details.

```python
from paddle.io import Dataset

class MyDataset(Dataset):
    """
    step 1：import the paddle.io.Dataset
    """
    def __init__(self, mode='train'):
        """
       step 2：implement the constructor, define the data reading method, and divide the training and test data sets
        """
        super().__init__()

        if mode == 'train':
            self.data = [
                ['traindata1', 'label1'],
                ['traindata2', 'label2'],
                ['traindata3', 'label3'],
                ['traindata4', 'label4'],
            ]
        else:
            self.data = [
                ['testdata1', 'label1'],
                ['testdata2', 'label2'],
                ['testdata3', 'label3'],
                ['testdata4', 'label4'],
            ]

    def __getitem__(self, index):
        """
      step 3：implement __getitem__ method, define how to get data when index is specified, and return a single piece of data (training data, corresponding label)
        """
        data = self.data[index][0]
        label = self.data[index][1]

        return data, label

    def __len__(self):
        """
        step 4：implement __len__ method to return the total number of data sets
        """
        return len(self.data)

# Test the defined data set
train_dataset = MyDataset(mode='train')
val_dataset = MyDataset(mode='test')

print('=============train dataset=============')
for data, label in train_dataset:
    print(data, label)

print('=============evaluation dataset=============')
for data, label in val_dataset:
    print(data, label)
```

### Networking method
#### Sequential networking

Sequential can be used directly for sequential linear network structure to quickly complete the network, which can reduce the definition of classes and other code writing.

```python
import paddle

# Sequential networking
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)
```

#### SubClass networking

For some complex network structures, the model code can be written by using the Layer subclass definition, declaring the Layer in the `__init__` constructor and calculating with the declared Layer variables in the `forward`. The subclass networking approach also allows for the reuse of sublayer, which can be defined once in the constructor for the same layer and called multiple times in `forward`.

```python
import paddle

# Layer class networking
class Mnist(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

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

mnist = Mnist()
```

### Model training

#### High-level API

The `paddle.Model` high-level API has been added to simplify training, evaluation, and prediction class code development. Note the distinction between Model and Net concepts. Net refers to a network structure that inherits paddle.nn.Layer; while Model refers to a trainable, evaluation, and prediction instance that holds a Net object while specifying loss functions, optimization algorithms, and evaluation metrics. Refer to the code example of high-level API for details.

```python
import paddle
from paddle.vision.transforms import ToTensor

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())
lenet = paddle.vision.models.LeNet()

# Mnist inherits paddle.nn.Layer belonging to Net, model contains the training function
model = paddle.Model(lenet)

# set the optimizer, loss, metric for training model
model.prepare(
    paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy()
    )

# training
model.fit(train_dataset, epochs=2, batch_size=64, log_freq=200)

# evaluation
model.evaluate(test_dataset, log_freq=20, batch_size=64)
```

#### Basic API

```python
import paddle
from paddle.vision.transforms import ToTensor

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())
lenet = paddle.vision.models.LeNet()
loss_fn = paddle.nn.CrossEntropyLoss()

# load the training set batch_size as 64
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

def train():
    epochs = 2
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=lenet.parameters())
    # Use Adam as an optimization function
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = lenet(x_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss = loss_fn(predicts, y_data)
            loss.backward()
            if batch_id % 100 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            adam.step()
            adam.clear_grad()

# training
train()
```

### Stand-alone multi-card launch
`paddle.distributed.spawn` function is added to start standalone multi-card training, while the original `paddle.distributed.launch` method is still retained.

#### Method 1. launch

##### High-level API scenario

When `paddle.Model` high level is used to implement training, it is very simple to launch single-computer multi-card training, the code does not need any modification, only need to add the parameter `-m paddle.distributed.launch` when launching.

```bash
# launch single-computer multi-card training, use card 0 by default
$ python train.py

# launch single-computer multi-card training, use all visible card by default
$ python -m paddle.distributed.launch train.py

# launch single-computer multi-card training, set the card 0 and card 1
$ python -m paddle.distributed.launch --selected_gpus='0,1' train.py

# launch single-computer multi-card training, set the card 0 and card 1
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m paddle.distributed.launch train.py
```

##### Basuc API scenario

If basic API is used to implement stand-alone multi-card training, you need to make 3 changes, as follows.

```python
import paddle
from paddle.vision.transforms import ToTensor

# The 1st change, import the packages needed for distributed training
import paddle.distributed as dist

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())
lenet = paddle.vision.models.LeNet()
loss_fn = paddle.nn.CrossEntropyLoss()

# Load the training set batch_size as 64
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

def train(model):
    # The 2nd change, initialize parallel environment
    dist.init_parallel_env()

    # The 3rd change, add paddle.DataParallel wrapper
    lenet = paddle.DataParallel(model)
    epochs = 2
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=lenet.parameters())
    # Use Adam as the optimization function
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = lenet(x_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss = loss_fn(predicts, y_data)
            loss.backward()
            if batch_id % 100 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            adam.step()
            adam.clear_grad()

# Start training
train(lenet)
```

Save the file after modifying it, and then use the same startup method as the high-level API

**Note:** Single card training does not support  ``init_parallel_env``, please use the following ways for distributed training.

```bash

# launch single-computer multi-card training, use all visible card by default
$ python -m paddle.distributed.launch train.py

# launch single-computer multi-card training, set the card 0 and card 1
$ python -m paddle.distributed.launch --selected_gpus '0,1' train.py

# launch single-computer multi-card training, set the card 0 and card 1
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m paddle.distributed.launch train.py
```

#### Method 2. Spawn launch

Launch method need to start multiple processes as a file. You need to call ``paddle.distributed.launch`` at launch time with a higher process management requirements. PaddlePaddle framework 2.0 added ``spawn`` launch method, for better controlling the process and more friendly in the log printing and training exit. Usage examples are as follows.

```python
import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist

class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))

def train(print_result=False):

    # 1. initialize the parallel training environment
    dist.init_parallel_env()

    # 2. create parallel training Layer and Optimizer
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)

    loss_fn = nn.MSELoss()
    adam = opt.Adam(
        learning_rate=0.001, parameters=dp_layer.parameters())

    # 3. run the network
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    if print_result is True:
        print("loss:", loss.numpy())

    loss.backward()

    adam.step()
    adam.clear_grad()

# Usage 1: Only import the training function
# Applicable scenario: no parameters are required and all currently visible GPU devices are needed.
if __name__ == '__main__':
    dist.spawn(train)

# Usage 2：Import the training function and parameters
# Applicable scenario: some parameters are required and all currently visible GPU devices are needed.
if __name__ == '__main__':
    dist.spawn(train, args=(True,))

# Usage 3：Import the training function and parameters and specify the number of parallel processes
# Applicable scenario: some parameters are required and only part of the visible GPU devices are needed：
# The current machine has 8 GPU cards {0,1,2,3,4,5,6,7},  the first two cards {0,1} will be used.
# Or the current machine can make only 4 GPU cards visible by configuring the environment variable CUDA_VISIBLE_DEVICES=4,5,6,7
# GPU cards are visible, the first two visible cards will be used {4,5}
if __name__ == '__main__':
    dist.spawn(train, args=(True,), nprocs=2)

# Usage 4: Import the training function and parameters and specify the number of parallel processesand specify the card number currently in use
# Applicable scenario: some parameters are required and part of the visible GPU devices are needed, yet：
# You may not have the right to configure environment variables for the current machine due to permission issues, for example,the current machine has 8 GPU cards
# {0,1,2,3,4,5,6,7}, but you do not have the right to configure CUDA_VISIBLE_DEVICES, then you can
# Specify the parameter selected_gpus to select the card you wish to use, e.g. selected_gpus='4,5'.
# You can specify the use of card #4 and card #5
if __name__ == '__main__':
    dist.spawn(train, nprocs=2, selected_gpus='4,5')

# Usage 5: Specify the starting port for multi-card communication
# Applicable scenario: port establishment communication prompt requires retry or communication establishment failure
# By default, Paddle will find a free port on the current machine for multi-card communication, but
# When the machine environment is more complex, the port found by the program maybe unstable, so you can
# specify a stable free starting port for a better training experience
if __name__ == '__main__':
    dist.spawn(train, nprocs=2, started_port=12345)
```

### Model saving
Two formats: the training format, which saves model parameters and optimizer-related state and can be used to resume training; the prediction format, which saves the predicted static graph network structure as well as parameters for prediction deployment.
#### High-level API scenario

The model preservation methods used for prediction deployment are as follows:

```python
model = paddle.Model(Mnist())
# Prediction format, saved models can be used for prediction deployment
model.save('mnist', training=False)
# You can  get the model needed for predictive deployment
```

#### Basic API Scenarios

Dynamic graph training models can be converted to deployable static graph models through the dynamic conversion function as follows:

```python
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    # The first change
    # Specify the shape of the input data via InputSpec, None means variable length
    # Convert dynamic graph to static graph Program by to_static decorator
    @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out


net = SimpleNet()

# The second change
# Save static graph model that can be used for predictive deployment
paddle.jit.save(net, './simple_net')
```
### Inference
The inference library of Paddle Inference API has been upgraded to simplify the writing style and to remove historically redundant concepts. The original API remains unchanged. The new API system is recommended, and the old API will be gradually removed in subsequent versions.

#### C++ API

Important changes：

- Namespace changed from `paddle` to `paddle_infer`

- `PaddleTensor`, `PaddleBuf`, etc. are deprecated, `ZeroCopyTensor` becomes the default Tensor type and renamed to `Tensor`.

- New `PredictorPool` tool class to simplify the creation of multi-threaded predictors, and more peripheral tools will be added

- Return value of `CreatePredictor` (formerly `CreatePaddlePredictor`) changed from `unique_ptr` to `shared_ptr` to avoid the problem of wrong decomposition order after Clone



API changes

| Original name                 | Present name                 | Changes                                                   |
| ----------------------------- | ---------------------------- | --------------------------------------------------------- |
| Header files `paddle_infer.h` | Unchanged                    | Include old interfaces to maintain backward compatibility |
| None                          | `paddle_inference_api.h`     | New API that can co-exist with the old interface          |
| `CreatePaddlePredictor`       | `CreatePredictor`            | Return value becomes shared_ptr                           |
| `ZeroCopyTensor`              | `Tensor`                     | None                                                      |
| `AnalysisConfig`              | `Config`                     | None                                                      |
| `TensorRTConfig`              | Abandoned                    |                                                           |
| `PaddleTensor` + `PaddleBuf`  | Abandoned                    |                                                           |
| `Predictor::GetInputTensor`   | `Predictor::GetInputHandle`  | None                                                      |
| `Predictor::GetOutputTensor`  | `Predictor::GetOutputHandle` | None                                                      |
|                               | `PredictorPool`              | Simplify support for creating multiple predictors         |

The process for using the new C++ API is exactly the same as before, with only naming changes.

```c++
#include "paddle_infernce_api.h"
using namespace paddle_infer;

Config config;
config.SetModel("xxx_model_dir");

auto predictor = CreatePredictor(config);

// Get the handles for the inputs and outputs of the model
auto input0 = predictor->GetInputHandle("X");
auto output0 = predictor->GetOutputHandle("Out");

for (...) {
  // Assign data to input0
  MyServiceSetData(input0);

  predictor->Run();

  // get data from the output0 handle
  MyServiceGetData(output0);
}
```

#### Python API

Python API changes are correspond to C++ and will be released in version 2.0.


## Appendix
### 2.0 Conversion Tool
In order to downgrade the upgrade cost, a conversion tool is provided to upgrade the code developed in Paddle 1.8  to API 2.0. The API 2.0  has been upgraded in API name, parameter name, behavior, etc. The conversion tool currently cannot cover all the API upgrades; for APIs that cannot be converted, the conversion tool will report an error and prompt for a manual upgrade.

https://github.com/PaddlePaddle/paddle_upgrade_tool

For APIs not covered by the conversion tool, please check the API documentation on the official website and manually upgrade the code's API.

#### 2.0 Documentation Tutorials
Some sample tutorials for version 2.0 are provided below:

You can check the [Application Practice](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/tutorial/index_cn.html) section of the official website, or download the source code provided here:

https://github.com/PaddlePaddle/book/tree/develop/paddle2.0_docs
