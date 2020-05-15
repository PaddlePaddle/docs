..  _user_guides_dataloader_using:

#################
数据准备、载入及加速
#################

在命令式编程模式（动态图）下，使用PaddlePaddle Fluid准备数据相比声明式编程模式（静态图）较为简洁，总体分为三个步骤：

Step 1: 自定义Reader生成训练/预测数据
###################################

首先，自定义一个生成器类型的Reader数据源，通过yield的方式顺序输出训练/预测数据，其生成的数据类型可以为Numpy Array或Tensor。

根据Reader返回的数据形式的不同，Fluid支持配置三种不同的数据生成器，以满足不同的用户需求，下面举例说明。

以常见的图像类模型输入需求为例，一个Sample的输入通常是 (image, label)，即（处理图像，真实标签），对此Fluid支持的三种不同Reader分别为：

1. Sample（样本）级的Reader：每次生成的数据为 (image, label)
2. Sample List 级的Reader：每次生成的数据为 [(image, label), (image, label), (image, label), ...]
3. Batch 级的Reader：每次生成的数据为 ([BATCH_SIZE, image], [BATCH_SIZE, label])

如果您的数据是Sample形式的数据，但是想要在外部以Batch为单位组件数据，并对整个Batch的数据进行预处理，我们提供了相关的工具，详细内容请参见： `数据预处理工具 <../static_mode/reader_cn.html>`_ ，此处仅通过简单的例子说明如下：


```python
import paddle

mnist_train = paddle.dataset.mnist.train()
mnist_train_batch_reader = paddle.batch(mnist_train, 128) # 128 为 batch size
```

在上面例子中，mnist_train生成的数据是Sample为单位的，经过 :code:`paddle.batch` 处理后，会以一个Batch（包含128个Sample）为单位生成数据。


Step 2. 创建DataLoader并设置自定义Reader
######################################

在命令式编程模式（动态图）下，我们推荐使用DataLoader进行数据载入，DataLoader默认使用线程进行异步加速，使用也更加简便直观。

1. 创建DataLoader

命令式编程模式（动态图）下，创建DataLoader对象的方式为：

.. code-block:: python

    import paddle.fluid as fluid

    data_loader = fluid.io.DataLoader.from_generator(capacity=10)

其中，

- capacity为DataLoader对象的缓存区大小，单位为batch数量。

该值根据实际需求设定即可，一般对于单卡训练的场景，设置5到20即可满足需求，特别是对于一个Batch数据较大的情况，不宜设置过大，可能会消耗过多内存资源，反而降低效率。
如果模型消耗训练或预测数据较慢的话，缓存队列设置过大也没有收益，但对于模型训练速度非常快，数据载入是训练速度瓶颈的情况，适当扩大队列能够提高训练速度。

对于其他参数，我们推荐使用默认设置，无需再额外配置，具体可参见官方文档 :ref:`cn_api_fluid_io_DataLoader` 

2. 设置DataLoader对象的数据源

上文中讲到，DataLoader支持配置三种不同的自定义Reader，配置这三种Reader的方法分别为： :code:`set_sample_generator()` ， :code:`set_sample_list_generator` 和 :code:`set_batch_generator()` 。
这三个方法均接收Python生成器 :code:`generator` 作为参数，其区别在于：

- :code:`set_sample_generator()` 要求 :code:`generator` 返回的数据格式为(image_1, label_1)，其中image1和label_1为单个样本的Numpy Array类型数据。

- :code:`set_sample_list_generator()` 要求 :code:`generator` 返回的数据格式为[(image_1, label_1), (image_2, label_2), ..., (image_n, label_n)]，其中image_i和label_i均为每个样本的Numpy Array类型数据，n为batch size。

- :code:`set_batch_generator()` 要求 :code:`generator` 返回的数据的数据格式为([BATCH_SIZE, image], [BATCH_SIZE, label])，其中[BATCH_SIZE, image]和[BATCH_SIZE, label]为batch级的Numpy Array或Tensor类型数据。

此处我们构建三个不同的示例生成器，对应上述三个接口：

.. code-block:: python

    import numpy as np

    BATCH_NUM = 10
    BATCH_SIZE = 16
    MNIST_IMAGE_SIZE = 784
    MNIST_LABLE_SIZE = 1

    # 伪数据生成函数，服务于下述三种不同的生成器
    def get_random_images_and_labels(image_shape, label_shape):
        image = np.random.random(size=image_shape).astype('float32')
        label = np.random.random(size=label_shape).astype('int64')
        return image, label

    # 每次生成一个Sample，使用set_sample_generator配置数据源
    def sample_generator_creator():
        def __reader__():
            for _ in range(BATCH_NUM * BATCH_SIZE):
                image, label = get_random_images_and_labels([MNIST_IMAGE_SIZE], [MNIST_LABLE_SIZE])
                yield image, label

        return __reader__

    # 每次生成一个Sample List，使用set_sample_list_generator配置数据源
    def sample_list_generator_creator():
        def __reader__():
            for _ in range(BATCH_NUM):
                sample_list = []
                for _ in range(BATCH_SIZE):
                    image, label = get_random_images_and_labels([MNIST_IMAGE_SIZE], [MNIST_LABLE_SIZE])
                    sample_list.append([image, label])

                yield sample_list

        return __reader__

    # 每次生成一个Batch，使用set_batch_generator配置数据源
    def batch_generator_creator():
        def __reader__():
            for _ in range(BATCH_NUM):
                batch_image, batch_label = get_random_images_and_labels([BATCH_SIZE, MNIST_LABLE_SIZE], [BATCH_SIZE, MNIST_LABLE_SIZE])
                yield batch_image, batch_label

        return __reader__

然后，可以根据需求为DataLoader配置不同的数据源，此处完整的创建DataLoader及相应配置为：

.. code-block:: python

    import paddle.fluid as fluid

    BATCH_SIZE = 16

    place = fluid.CPUPlace() # 或者 fluid.CUDAPlace(0)
    fluid.enable_imperative(place)

    # 使用sample数据生成器作为DataLoader的数据源
    data_loader1 = fluid.io.DataLoader.from_generator(capacity=10)
    data_loader1.set_sample_generator(sample_generator_creator(), batch_size=BATCH_SIZE, places=place)

    # 使用sample list数据生成器作为DataLoader的数据源
    data_loader2 = fluid.io.DataLoader.from_generator(capacity=10)
    data_loader2.set_sample_list_generator(sample_list_generator_creator(), places=place)

    # 使用batch数据生成器作为DataLoader的数据源
    data_loader3 = fluid.io.DataLoader.from_generator(capacity=10)
    data_loader3.set_batch_generator(batch_generator_creator(), places=place)


此处有两点值得注意：

1. DataLoader的使用需要在命令式编程模式（动态图）下，即提前通过 :code:`fluid.enable_imperative()` 进入命令式编程模式（动态图）。
2. 命令式编程模式（动态图）下DataLoader配置数据源，需要在 :code:`set_XXX_generator` 时执行place，一般为当前执行的place。（该点后续可能会优化，在这里默认使用动态图当前place，而无需用户指定）


Step 3. 使用DataLoader进行模型训练和预测
####################################

下面我们通过一个完整的例子来说明命令式编程模式（动态图）下DataLoader在训练/预测时的使用：

1. 构建命令式编程模式（动态图）模型

此处我们构建一个简单的命令式编程模式（动态图）网络。

2. 创建网络执行对象，配置DataLoader，进行训练或预测

.. code-block:: python

    import paddle.fluid as fluid

    EPOCH_NUM = 4
    BATCH_SIZE = 16
    MNIST_IMAGE_SIZE = 784
    MNIST_LABLE_SIZE = 1

    # 1. 构建命令式编程模式（动态图）网络
    class MyLayer(fluid.dygraph.Layer):
        def __init__(self):
            super(MyLayer, self).__init__()
            self.linear = fluid.dygraph.nn.Linear(MNIST_LABLE_SIZE, 10)

        def forward(self, inputs, label=None):
            x = self.linear(inputs)
            if label is not None:
                loss = fluid.layers.cross_entropy(x, label)
                avg_loss = fluid.layers.mean(loss)
                return x, avg_loss
            else:
                return x

    # 2. 创建网络执行对象，配置DataLoader，进行训练或预测
    place = fluid.CPUPlace() # 或者 fluid.CUDAPlace(0)
    fluid.enable_imperative(place)

    # 创建执行的网络对象
    my_layer = MyLayer()

    # 添加优化器
    adam = fluid.optimizer.AdamOptimizer(
        learning_rate=0.001, parameter_list=my_layer.parameters())

    # 配置DataLoader
    train_loader = fluid.io.DataLoader.from_generator(capacity=10)
    train_loader.set_sample_list_generator(sample_list_generator_creator(), places=place)
    
    # 执行训练/预测
    for _ in range(EPOCH_NUM):
        for data in train_loader():
            # 拆解载入数据
            image, label = data

            # 执行前向
            x, avg_loss = my_layer(image, label)

            # 执行反向
            avg_loss.backward()

            # 梯度更新
            adam.minimize(avg_loss)
            mnist.clear_gradients()


异步数据读取加速
##############

在命令式编程模式（动态图）下，DataLoader默认使用子线程进行异步数据读取加速，但由于python GIL（全局解释器锁）的限制，在数据载入开销比较大的场景下，仅使用线程进行加速的效果并不能满足训练速度需求，因此我们提供了使用子进程加速的方式，进一步提升数据读取的效率。

配置使用子进程加速，仅需要在DataLoader创建时设置 :code:`use_multiprocess=True` 即可，此参数默认为False，例如

.. code-block:: python

    import paddle.fluid as fluid

    data_loader = fluid.io.DataLoader.from_generator(capacity=10, use_multiprocess=True)

其他使用方式均与前文中的示例一致。

关于配置此选项带来的加速效果，此处列出一些测试数据供参考。表中数据为单个Epoch的训练耗时，单位为秒(s)，模型名后括号内为模型训练所使用的Batch Size。

.. list-table:: 
   :widths: 25 25 25 25
   :header-rows: 1

   * - 模型 (Batch Size)
     - DataLoader默认模式耗时 (s)
     - DataLoader+子进程模式耗时 (s)
     - 加速比例
   * - Mnist (64)
     - 10.16 
     - 6.44 
     - **+57.8%**
   * - ResNet (32)
     - 83.56
     - 53.75 
     - **+55.5%**
   * - Ptb (20)
     - 108.56
     - 108.27 
     - **+0.2%**
   * - MobileNet V1 (256)
     - 5041.77
     - 3249.51 
     - **+55.2%**

从表中可以看出，在图像类训练任务这种数据载入开销较大的情况下，配置子进程加速的效果是比较明显的，但在一个Batch数据很小的训练任务中，没有明显提升，在这种情况下，数据载入的开销不会是训练速度的瓶颈，使用默认模式即可。

.. note::
    命令式编程模式（动态图）下DataLoader多进程方式采用共享内存机制实现Tensor的进程间传输，使用时需要保证机器上或Docker共享内存空间足够大，需要大于 :code:`DataLoader.capacity * Batch Size * 单个Sample的数据大小`。在物理机或者虚拟机上训练一般不会有问题，但在docker中可能出现共享内存不足的情况，因为docker中/dev/shm目录空间不够大（默认64M），若内存空间不足，建议配置较大的共享内存空间，或者仍使用单进程。