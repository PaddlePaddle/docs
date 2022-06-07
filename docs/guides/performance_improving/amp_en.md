# Automatic Mixed Precision Training

In general, the datatype of training deep learning models is single-precision floating-point format(also called FP32). In 2018, Baidu and NVIDIA jointly published the paper: [MIXED PRECISION TRAINING](https://arxiv.org/pdf/1710.03740.pdf), which proposed mixed precision training. During the process of training, some operators use FP32 and other operators use half precision(also called FP16) in the same time. Its purpose is to speed up training, while compared with the FP32 training model, the same accuracy is maintained. This tutorial will introduce how to use automatic mixed precision training with PaddlePaddle.  

## I. overview

### 1.1. Half Precision (FP16)

First introduce FP16. As shown in Figure 1, FP16 occupies 16 bits (two bytes in modern computers) of computer memory. In the IEEE 754-2008 standard, it is also named binary16. Compared with FP32 and double precision (also called FP64) commonly used, FP16 is more suitable for the usage in scenarios with low precision requirements.

<figure align="center">
    <img src="https://paddleweb-static.bj.bcebos.com/images/fp16.png" width="600" alt='missing'/>
    <figcaption><center>Figure 1. Half precision(FP16) and single precision(FP32)</center></figcaption>
</figure>

### 1.2. FP16 Computing Power of NVIDIA GPU

When the same hyperparameters are used, mixed precision training using FP16 and FP32 can achieve the same accuracy as that of pure single precision used, and can accelerate the training speed. It mainly attributes to the features that NVIDIA Volta and NVIDIA Turing use FP16 to calculate:
- FP16 can reduce memory bandwidth and storage requirements by half, which allows researchers to use more complex models and larger batch sizes under the same hardware conditions.
- FP16 can make full use of Tensor Cores technology provided by NVIDIA Volta and NVIDIA Turing. On the same GPU hardware, the computing throughput of Tensor Cores' FP16 is 8 times bigger than that of FP32.

The ``nvidia-smi`` command can help you view NVIDIA GPU architecture information. In addition, if the amp training mode is enabled, PaddlePaddle will automatically help detect whether the hardware environment meets the above hardware conditions. If not, the following warning messages will be provided: ``UserWarning: AMP only support NVIDIA GPU with Compute Capability 7.0 or higher, current GPU is: Tesla K40m, with Compute Capability: 3.5.``.

## II. Automatic Mixed Precision Training with PaddlePaddle

Using PaddlePaddle's API ``paddle.amp.auto_cast`` and ``paddle.amp.GradScaler`` can realize automatic mixed precision training (AMP), which can automatically choose FP16 or FP32 for different operators' calculation. According to the use degree of FP16 in the model, the AMP is divided into two levels:
- level = ’O1‘: The black&white operator list strategy is used for AMP. The op in the black list will be calculated by FP32, and the op in the white list will be calculated by FP16. During the training process, Paddle will automatically change the input data type of the op in the white list from FP32 to FP16. The operator list calculated by FP16 and FP32 can be found in this [document](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/Overview_cn.html).For an op that is not in the black&white list, Paddle will infer based on all the input data types of the op. when all the inputs are FP16, the op will directly use FP16 for calculation, otherwise FP32 for calculation.
- level = ’O2‘: This mode adopts a more radical strategy than O1. Except ops that the Paddle does not support calculated by FP16, all other ops use FP16. Paddle will cast the neural network parameters from FP32 to FP16. Compared with O1, the training speed will be significantly improved, but there may be accuracy problems. Therefore, Paddle provides a user-defined blacklist through which you can specify some ops with accuracy problems to perform FP32 operations.

Both the Dynamic and Static Graph of the Paddle provide users with convenient APIs to enable the AMP training. The following takes the specific training code as an example to learn how to use Paddle to realize AMP training.

### 2.1. AMP in Dynamic Graph

Paddle Dynamic Graph provides a series of convenient APIs for AMP: ``paddle.amp.auto_cast``, ``paddle.amp.GradScaler``, ``paddle.amp.decorate``。 Of which:

1) ``paddle.amp.auto_cast``: used to create a context environment of AMP to support the AMP strategy of ops executed in Dynamic Graph.

2) ``paddle.amp.GradScaler``: GradScaler is used to train "AMP" in the Dynamic Graph mode. It can control the scaling ratio of loss and help avoid floating-point overflow (Note: optional, if FP16 data type is used to ensure that parameters will not overflow, it is not necessary to call this interface)

3) ``paddle.amp.decorate``: It is used to cast the neural network parameter data type to FP16 (except BatchNorm and LayerNorm) in ``O2`` (Note: in ``O1`` mode, this interface has no function and does not need to be called)

<a name="2.1.1"></a>

#### 2.1.1. FP32 training mode of Dynamic Graph

1) Build a simple neural network by Paddle: it is used to compare the training speed of using FP32 training and using AMP training. In order to fully reflect the speed improvement brought by AMP, build a network composed of nine layers of ``linear``. Each layer of ``linear`` network is composed of ``matmul`` and ``add`` operator. The ``matmul`` is an operator that can be accelerated.

```python
import time
import paddle
import paddle.nn as nn
import numpy

paddle.seed(100)
numpy.random.seed(100)
place = paddle.CUDAPlace(0)

class SimpleNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.linears = paddle.nn.LayerList(
            [paddle.nn.Linear(input_size, output_size) for i in range(9)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i](x)
        return x
```
2) Set relevant training parameters and training data: in order to effectively see the improvement of training speed by AMP, set the value of ``input_size`` and ``output_size`` to a larger value. In order to use the ``tensor core`` provided by the GPU, it is also necessary to set ``batch_size`` a multiple of 8 (for the performance optimization method based on mixed accuracy training, see: <a href="#III">III. AMP performance optimization</a>）。

```python
epochs = 2
input_size = 8192  
output_size = 8192  
batch_size = 2048
nums_batch = 10

from paddle.io import Dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        data = numpy.random.random([input_size]).astype('float32')
        label = numpy.random.random([output_size]).astype('float32')
        return data, label

    def __len__(self):
        return self.num_samples

dataset = RandomDataset(nums_batch * batch_size)
loader = paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
```

Note: if the sample code shows an error related to out of memory on your machine, please try setting ``input_size``、``output_size``、``batch_size `` decrease.

3) Using Dynamic Graph FP32 training:

```python
mse = paddle.nn.MSELoss()
model = SimpleNet(input_size, output_size)  
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

train_time = 0
for epoch in range(epochs):
    for i, (data, label) in enumerate(loader):
        start_time = time.time()
        label._to(place)
        # forward
        output = model(data)
        loss = mse(output, label)
        # backward
        loss.backward()
        # update parameters
        optimizer.step()
        optimizer.clear_grad(set_to_zero=False)

        train_loss = loss.numpy()
        train_time += time.time() - start_time

print("loss:", train_loss)
print("Time consuming using FP32 mode:{:.3f} sec".format(train_time/(epochs*nums_batch)))
```

    loss: [0.6486028]
    Time consuming using FP32 mode:0.529 sec

#### 2.1.2. AMP-O1 training mode of Dynamic Graph

In Paddle, two logics need to be added on the basis of FP32 code when AMP-O1 is used for training:

- logic 1: use ``paddle.amp.auto_cast`` to create a context environment of AMP. In AMP context, Paddle will automatically determine the input data type (FP16 or FP32) of each OP according to the black&white list preset by Paddle.
- logic 2: optional, use ``paddle.amp.Gradscaler`` to scale the ``loss`` to avoid floating-point underflow (underflow: the valid data representation range of FP16 is [6.10×10−5, 65504] smaller than FP32, gradient values of smaller parameters in the model may not be represented in FP16)

```python
mse = paddle.nn.MSELoss()
model = SimpleNet(input_size, output_size)  
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  

# logic 2: optional, use ``paddle.amp.Gradscaler`` to scale the ``loss`` to avoid floating-point underflow
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

train_time = 0
for epoch in range(epochs):
    for i, (data, label) in enumerate(loader):
        start_time = time.time()

        label._to(place)
        # logic 1: create a context environment of AMP
        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
            output = model(data)
            loss = mse(output, label)
        # logic 2: use GradScaler to scale the loss
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(optimizer)  
        scaler.update()
        optimizer.clear_grad(set_to_zero=False)

        train_loss = loss.numpy()
        train_time += time.time() - start_time

print("loss:", train_loss)
print("Time consuming using AMP-O1 mode:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```
In the above code, in ``paddle.amp.auto_cast`` context, the ``model`` and ``mse`` are executed by the logic of AMP-O1, because ``elementwise_add`` is added to the white list so that the ``matmul`` operator and ``add`` operator of the linear layer perform FP16 calculation.

    loss: [0.6486219]
    Time consuming using AMP-O1 mode:0.118 sec

- ``paddle.amp.GradScaler`` introduction: [API doc](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/amp/GradScaler_en.html)
- ``paddle.amp.auto_cast`` introduction: [API doc](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/amp/auto_cast_en.html)

#### 2.1.3. AMP-O2 training mode of Dynamic Graph

O2 mode adopts a more radical strategy than O1. Except ops that the Paddle does not support calculated by FP16, all other ops use FP16. The network parameters need to be casted from FP32 to FP16 before training. Add three logics on the basis of FP32 code:

- logic 1: use ``paddle.amp.decorate`` cast network parameters from FP32 to FP16.
- logic 2: use ``paddle.amp.auto_cast`` to create a context environment of AMP，Paddle will use FP16 to calculate all ops except the customized blacklist.
- logic 3: optional, use ``paddle.amp.Gradscaler`` to scale the ``loss`` to avoid floating-point underflow (underflow: the valid data representation range of FP16 is [6.10×10−5, 65504] smaller than FP32, gradient values of smaller parameters in the model may not be represented in FP16)

```python
mse = paddle.nn.MSELoss()
model = SimpleNet(input_size, output_size)
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

# logic 1: cast network parameters from FP32 to FP16.
model = paddle.amp.decorate(models=model, level='O2')

# logic 3: optional,  use GradScaler to scale the ``loss`` to avoid floating-point underflow
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

train_time = 0
for epoch in range(epochs):
    for i, (data, label) in enumerate(loader):
        start_time = time.time()

        label._to(place)
        # logic 2: create a context environment of AMP
        with paddle.amp.auto_cast(level='O2'):
            output = model(data)
            loss = mse(output, label)
         # logic 3: use GradScaler to scale the loss
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(optimizer)  
        scaler.update()  
        optimizer.clear_grad(set_to_zero=False)

        train_loss = loss.numpy()
        train_time += time.time() - start_time

print("loss=", train_loss)
print("Time consuming using AMP-O2 mode:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```

    loss= [0.6743]
    Time consuming using AMP-O2 mode:0.102 sec

- ``paddle.amp.decorate`` introduction: [API doc](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/amp/decorate_en.html)

The comparison of the accuracy and speed of FP32 and AMP training is shown in the following table:

|test | FP32 | AMP-O1 | AMP-O2 |
|:---:|:---:|:---:|:---:|
|Time consuming | 0.529s | 0.118s | 0.102s |
|loss | 0.6486028 | 0.6486219 | 0.6743 |

It can be seen from the statistical results in the above table that the training speed in O1 mode is increased by about 4.5 times, and that in O2 mode is increased by about 5.2 times. For more examples of using hybrid accuracy training, please refer to the Paddle model library: [paddlepaddle/models](https://github.com/PaddlePaddle/models).

Note: due to the machine environment, the training time statistics of the above example codes may be different. The impact mainly includes: GPU utilization, CPU utilization, etc. the test machine configuration is as follows:

|Device | MEM Clocks | SM Clocks | Running with CPU Clocks |
|:---:|:---:|:---:|:---:|
|Tesla V100 SXM2 16GB |  877 MHz   | 1530 MHz |   1000 - 2400 MHz  |

### 2.2. AMP in Static Graph

Paddle Static Graph provides a series of convenient APIs for AMP: ``paddle.static.amp.decorate``, ``paddle.static.amp.fp16_guard``.

#### 2.2.1. FP32 training mode of Static Graph

Adopt the same network structure as Dynamic Graph training in section 2.1.1: <a href="#2.1.1">2.1.1 FP32 training mode of Dynamic Graph</a>

```python
paddle.enable_static()
place = paddle.CUDAPlace(0)
main_program = paddle.static.default_main_program()
startup_program = paddle.static.default_startup_program()

model = SimpleNet(input_size, output_size)
mse_loss = paddle.nn.MSELoss()
```

Static Graph training code is as follows:

```python
data = paddle.static.data(name='data', shape=[batch_size, input_size], dtype='float32')
label = paddle.static.data(name='label', shape=[batch_size, input_size], dtype='float32')

predict = model(data)
loss = mse_loss(predict, label)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())
optimizer.minimize(loss)

exe = paddle.static.Executor(place)
exe.run(startup_program)

train_time = 0
for epoch in range(epochs):
    for i, (train_data, train_label) in enumerate(loader()):
        start_time = time.time()
        train_loss = exe.run(main_program, feed={data.name: train_data, label.name: train_label }, fetch_list=[loss.name], use_program_cache=True)
        train_time += time.time() - start_time

print("loss:", train_loss)
print("Time consuming using FP32 mode:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```

    loss: [array([0.6486028], dtype=float32)]
    Time consuming using FP32 mode:0.531 sec

#### 2.2.2. AMP-O1 training mode of Static Graph

The Static Graph uses ``paddle.static.amp.decorate`` to decorate the optimizer and use ``paddle.static.amp.CustomOpLists`` to define the black&white list to start the AMP training. The example code is as follows:

```python
data = paddle.static.data(name='data', shape=[batch_size, input_size], dtype='float32')
label = paddle.static.data(name='label', shape=[batch_size, input_size], dtype='float32')

predict = model(data)
loss = mse_loss(predict, label)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

# 1) use `CustomOpLists` to define Black and white lists
amp_list = paddle.static.amp.CustomOpLists(custom_white_list=['elementwise_add'])

# 2）decorate the optimizer for amp:
optimizer = paddle.static.amp.decorate(
    optimizer=optimizer,
    amp_lists=amp_list,
    init_loss_scaling=128.0,
    use_dynamic_loss_scaling=True)

optimizer.minimize(loss)

exe = paddle.static.Executor(place)
exe.run(startup_program)

train_time = 0
for epoch in range(epochs):
    for i, (train_data, train_label) in enumerate(loader()):
        start_time = time.time()
        train_loss = exe.run(main_program, feed={data.name: train_data, label.name: train_label }, fetch_list=[loss.name], use_program_cache=True)
        train_time += time.time() - start_time

print("loss:", train_loss)
print("Time consuming using AMP-O1 mode:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```

    loss: [array([0.6486222], dtype=float32)]
    Time consuming using AMP-O1 mode:0.117 sec

`paddle.static.amp.CustomOpLists` is used to customize the black-and-white list. The black list OP implements FP32 kernel and the white list OP implements FP16 kernel.

#### 2.2.3. AMP-O2 training mode of Static Graph

There are two ways to start AMP-O2 in Static Graph:

- use ``paddle.static.amp.fp16_guard`` to control the scope of FP16 application. All ops in this context will perform FP16 calculation, and other OPS will perform FP32 calculation;

- not use``paddle.static.amp.fp16_guard`` to control the scope of FP16 application. All the ops of the network perform FP16 calculation (excluding the ops in the user-defined blacklist and those that do not support FP16 calculation)

1) Set ``paddle.static.amp.decorate`` parameter ``use_pure_fp16`` is True, and the parameter ``use_fp16_guard`` is False

```python
data = paddle.static.data(name='data', shape=[batch_size, input_size], dtype='float32')
label = paddle.static.data(name='label', shape=[batch_size, input_size], dtype='float32')

predict = model(data)
loss = mse_loss(predict, label)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

# 1）decorate the optimizer for amp:
optimizer = paddle.static.amp.decorate(
    optimizer=optimizer,
    init_loss_scaling=128.0,
    use_dynamic_loss_scaling=True,
    use_pure_fp16=True,
    use_fp16_guard=False)

optimizer.minimize(loss)

exe = paddle.static.Executor(place)
exe.run(startup_program)

# 2) use `amp_init` convert FP32 parameters of the network to FP16
optimizer.amp_init(place, scope=paddle.static.global_scope())

train_time = 0
for epoch in range(epochs):
    for i, (train_data, train_label) in enumerate(loader()):
        start_time = time.time()
        train_loss = exe.run(main_program, feed={data.name: train_data, label.name: train_label }, fetch_list=[loss.name], use_program_cache=True)
        train_time += time.time() - start_time

print("loss:", train_loss)
print("Time consuming using AMP-O2 mode:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```

    loss: [array([0.6743], dtype=float16)]
    Time consuming using AMP-O2 mode:0.098 sec

Note: in AMP-O2 mode, the network parameters will be changed from FP32 to FP16. The input data needs to be FP16 data type. Therefore, the data type initialized in the ``class randomdataset`` needs to be set to ``float16``.

2) Set ``paddle.static.amp.decorate`` parameter ``use_pure_fp16`` is True, and the parameter ``use_fp16_guard`` is true, and use ``paddle.static.amp.fp16_guard`` control the calculation range of FP16.

Add code to model definition `fp16_guard` control part of network execution under FP16:

```python
class SimpleNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.linears = paddle.nn.LayerList(
            [paddle.nn.Linear(input_size, output_size) for i in range(9)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            if i > 0:
                with paddle.static.amp.fp16_guard():
                    x = self.linears[i](x)
            else:
                x = self.linears[i](x)
        return x
```

The training codes in this mode are as follows:

```python
data = paddle.static.data(name='data', shape=[batch_size, input_size], dtype='float32')
label = paddle.static.data(name='label', shape=[batch_size, input_size], dtype='float32')

predict = model(data)
loss = mse_loss(predict, label)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

# 1）decorate the optimizer for amp:
optimizer = paddle.static.amp.decorate(
    optimizer=optimizer,
    init_loss_scaling=128.0,
    use_dynamic_loss_scaling=True,
    use_pure_fp16=True,
    use_fp16_guard=True)

optimizer.minimize(loss)

exe = paddle.static.Executor(place)
exe.run(startup_program)

# 2) use `amp_init` convert FP32 parameters of the network to FP16
optimizer.amp_init(place, scope=paddle.static.global_scope())

train_time = 0
for epoch in range(epochs):
    for i, (train_data, train_label) in enumerate(loader()):
        start_time = time.time()

        train_loss = exe.run(main_program, feed={data.name: train_data, label.name: train_label }, fetch_list=[loss.name], use_program_cache=True)

        train_time += time.time() - start_time

print("loss:", train_loss)
print("Time consuming using AMP-O2 mode:{:.3f} sec".format(train_time/(epochs*nums_batch)))

```

    loss: [array([0.6691731], dtype=float32)]
    Time consuming using AMP-O2 mode:0.140 sec

The comparison of accuracy and speed of Static Graph FP32 and AMP training is shown in the following table:

|test | FP32 | AMP-O1 | AMP-O2 |
|:---:|:---:|:---:|:---:|
|Time consuming | 0.531s | 0.117s | 0.098s |
|loss | 0.6486028 | 0.6486222 | 0.6743 |

It can be seen from the statistical results in the above table that the training speed in O1 mode is increased by about 4.5 times, and that in O2 mode is increased by about 5.2 times. For more examples of using hybrid accuracy training, please refer to the Paddle model library: [paddlepaddle/models](https://github.com/PaddlePaddle/models).


<a name="III"></a>
## III. AMP performance optimization

The fundamental reason why the Paddle AMP improves the training performance of the model is that: the Tensor Core is used to accelerate the ``matmul`` and ``conv`` under FP16. In order to obtain the best acceleration effect, the Tensor Core has certain use constraints on matrix multiplication and convolution operations. The constraints are as follows:

### 3.1. Suggestion for matrix multiplication

General matrix multiplication (GEMM) is defined as: ``C = A * B + C``, of which:
- The dimension of matrix A is: M x K
- The dimension of matrix B is: K x N
- The dimension of matrix C is: M x N

Suggestion for matrix multiplication is:
- According to the Tensor Core usage recommendations, when the matrix dimensions of M, N, and K are multiples of 8 (the A100 architecture GPU is 16) (FP16 data), the performance is optimal.

### 3.2. Suggestions for convolution

Convolution is defined as: ``NKPQ = NCHW * KCRS``, of which:
- N: batch size
- K: Number of output channels
- P: Number of output height
- Q: Number of output width
- C: Number of input channels
- H: Number of input height
- W: Number of input width
- R: Number of filter height
- S: Number of filter width

Suggestions for convolution are:
- The number of input and output channels（C/K) to be divisible by 8 (for FP16)（Cudnn7.6.3 and above will be automatically filled if it is not a multiple of 8）
- For the first layer of the network, setting the number of channels to 4 can obtain the best operation performance (NVIDIA provides a special implementation for the convolution of the first layer of the network, and the performance is better when using 4 channels)
- Set the tensor layout in memory to NHWC format (if NCHW format is input, the Tesor Core will be automatically converted to NHWC. When the input and output values are large, the cost of this conversion is often greater)

## IV. Advanced Usage
### 4.1 Gradient Accumulation

Gradient accumulation means running a configured number of steps without updating the model variables. Until certain steps, use the accumulated gradients to update the variables. In automatic mixed precision training, gradient accumulation is also supported, and the usage is as follows:

```python
mse = paddle.nn.MSELoss()
model = SimpleNet(input_size, output_size)  # define model
optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # define optimizer

accumulate_batchs_num = 10 # the batch numbers of gradients accumulation

# define GradScaler
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

train_time = 0
for epoch in range(epochs):
    for i, (data, label) in enumerate(loader):
        start_time = time.time() # get start time
        label._to(place)
         # create AMP context environment
        with paddle.amp.auto_cast(level='O1'):
            output = model(data)
            loss = mse(output, label)
        # use GradScaler complete the loss scaling
        scaled = scaler.scale(loss)
        scaled.backward()

        #  when the accumulated batch is accumulate_batchs_num, update the model parameters
        if (i + 1) % accumulate_batchs_num == 0:
            # update parameters
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad(set_to_zero=False)

        train_loss = loss.numpy()
        train_time += time.time() - start_time

print("loss:", train_loss)
print("Time consuming using AMP-O1 mode:{:.3f} sec".format(train_time/(epochs*nums_batch)))
```

    loss: [0.6602017]
    Time consuming using AMP-O1 mode:0.113 sec
