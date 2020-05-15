..  _user_guides_dataloader_using:

#################
数据准备、载入及加速
#################

在动态图模式下，使用PaddlePaddle Fluid准备数据相比静态图模式较为简洁，总体分为三个步骤：

Step 1: 自定义Reader生成训练/预测数据
###################################

首先，自定义一个生成器类型的Reader数据源，通过yield的方式顺序输出训练/预测数据，其生成的数据类型可以为Numpy Array或LoDTensor。

根据Reader返回的数据形式的不同，Fluid支持配置三种不同的数据生成器，以满足不同的用户需求，下面举例说明。

以常见的图像类模型输入需求为例，一个Sample的输入通常是 (image, label)，即（处理图像，真实标签），对此Fluid支持的三种不同Reader分别为：

1. Sample（样本）级的Reader：每次生成的数据为 (image, label)
2. Sample List 级的Reader：每次生成的数据为 [(image, label), (image, label), (image, label), ...]
3. Batch 级的Reader：每次生成的数据为 ([BATCH_SIZE, image], [BATCH_SIZE, label])

如果您的数据是Sample形式的数据，但是想要在外部转换为Batch形式的数据，我们提供了相关可以进行数据预处理和组建Batch的工具：具体请参见： `数据预处理工具 <../static_mode/reader_cn.html>`_ 。

Step 2. 创建DataLoader并设置自定义Reader
######################################

在动态图模式下，我们推荐使用DataLoader进行数据载入，DataLoader默认使用线程进行异步加速，使用也更加简便直观。

1. 创建DataLoader

动态图模式下，创建DataLoader对象的方式为：

.. code-block:: python

    import paddle.fluid as fluid

    data_loader = fluid.io.DataLoader.from_generator(capacity=32)

其中，

- capacity为DataLoader对象的缓存区大小，单位为batch数量；

对于其他参数，我们推荐使用默认设置，无需再额外配置，具体可参见官方文档 :ref:`cn_api_fluid_io_DataLoader` 

2. 设置DataLoader对象的数据源

上文中讲到，DataLoader支持配置三种不同的自定义Reader，配置这三种Reader的方法分别为： :code:`set_sample_generator()` ， :code:`set_sample_list_generator` 和 :code:`set_batch_generator()` 。
这三个方法均接收Python生成器 :code:`generator` 作为参数，其区别在于：

- :code:`set_sample_generator()` 要求 :code:`generator` 返回的数据格式为[img_1, label_1]，其中img_1和label_1为单个样本的Numpy Array类型数据。

- :code:`set_sample_list_generator()` 要求 :code:`generator` 返回的数据格式为[(img_1, label_1), (img_2, label_2), ..., (img_n, label_n)]，其中img_i和label_i均为每个样本的Numpy Array类型数据，n为batch size。

- :code:`set_batch_generator()` 要求 :code:`generator` 返回的数据的数据格式为[batched_imgs, batched_labels]，其中batched_imgs和batched_labels为batch级的Numpy Array或LoDTensor类型数据。

此处我们构建三个不同的示例生成器，对应上述三个接口：

.. code-block:: python

    import numpy as np

    BATCH_NUM = 10
    BATCH_SIZE = 16

    # 伪数据生成函数，服务于下述三种不同的生成器
    def get_random_images_and_labels(image_shape, label_shape):
        image = np.random.random(size=image_shape).astype('float32')
        label = np.random.random(size=label_shape).astype('int64')
        return image, label

    # 每次生成一个Sample，使用set_sample_generator配置数据源
    def sample_generator_creator():
        def __reader__():
            for _ in range(BATCH_NUM * BATCH_SIZE):
                image, label = get_random_images_and_labels([784], [1])
                yield image, label

        return __reader__

    # 每次生成一个Sample List，使用set_sample_list_generator配置数据源
    def sample_list_generator_creator():
        def __reader__():
            for _ in range(BATCH_NUM):
                sample_list = []
                for _ in range(BATCH_SIZE):
                    image, label = get_random_images_and_labels([784], [1])
                    sample_list.append([image, label])

                yield sample_list

        return __reader__

    # 每次生成一个Batch，使用set_batch_generator配置数据源
    def batch_generator_creator():
        def __reader__():
            for _ in range(BATCH_NUM):
                batch_image, batch_label = get_random_images_and_labels([BATCH_SIZE, 784], [BATCH_SIZE, 1])
                yield batch_image, batch_label

        return __reader__

然后，可以根据需求为DataLoader配置不同的数据源，此处完整的创建DataLoader及相应配置为：

.. code-block:: python

    import paddle.fluid as fluid

    place = fluid.CPUPlace() # 或者 fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        # 使用sample数据生成器作为DataLoader的数据源
        data_loader1 = fluid.io.DataLoader.from_generator(capacity=10)
        data_loader1.set_sample_generator(sample_generator_creator(), batch_size=32, places=place)

        # 使用sample list数据生成器作为DataLoader的数据源
        data_loader2 = fluid.io.DataLoader.from_generator(capacity=10)
        data_loader2.set_sample_list_generator(sample_list_generator_creator(), places=place)

        # 使用batch数据生成器作为DataLoader的数据源
        data_loader3 = fluid.io.DataLoader.from_generator(capacity=10)
        data_loader3.set_batch_generator(batch_generator_creator(), places=place)


此处有两点值得注意：

1. 动态图DataLoader的使用需要在动态图模式下，即在 :code:`with fluid.dygraph.guard()` 环境中，或者提前通过 :code:`fluid.dygraph.enable_dygraph()` 进入动态图模式。
2. 动态DataLoader配置数据源，需要在 :code:`set_XXX_generator` 时执行place，一般为动态图当前执行的place。（该点后续可能会优化，在这里默认使用动态图当place，而无需用户指定）


Step 3. 使用DataLoader进行模型训练和预测
####################################

下面我们通过一个完整的例子来说明动态图模式下DataLoader在训练/预测时的使用：

1. 构建动态图模型

此处我们构建一个简单的动态图网络。

.. code-block:: python

    import paddle
    import paddle.fluid as fluid

    class MyLayer(fluid.dygraph.Layer):
        def __init__(self):
            super(MyLayer, self).__init__()
            self.linear = fluid.dygraph.nn.Linear(784, 10)

        def forward(self, inputs, label=None):
            x = self.linear(inputs)
            if label is not None:
                loss = fluid.layers.cross_entropy(x, label)
                avg_loss = fluid.layers.mean(loss)
                return x, avg_loss
            else:
                return x

2. 创建网络执行对象，配置DataLoader，进行训练或预测

.. code-block:: python

    import paddle.fluid as fluid

    place = fluid.CPUPlace() # 或者 fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):

        # 创建执行的网络对象
        my_layer = MyLayer()

        # 添加优化器
        adam = fluid.optimizer.AdamOptimizer(
            learning_rate=0.001, parameter_list=my_layer.parameters())

        # 配置DataLoader
        train_loader = fluid.io.DataLoader.from_generator(capacity=10)
        train_loader.set_sample_list_generator(sample_list_generator_creator(), places=place)
        
        # 执行训练/预测
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

在动态图模式下，DataLoader默认使用子线程进行异步数据读取加速，但由于python GIL（全局解释器锁）的限制，在数据载入开销比较大的场景下，仅使用线程进行加速的效果差强人意。

因此我们提供了使用子进程加速的方式，进一步提升数据读取的效率。

配置使用子进程加速，仅需要在DataLoader创建时设置 :code:`use_multiprocess=True` 即可，此参数默认为False，例如

.. code-block:: python

    import paddle.fluid as fluid

    data_loader = fluid.io.DataLoader.from_generator(capacity=32, use_multiprocess=True)

其他使用方式均与前文中的示例一致。

关于配置此选项带来的加速效果，此处列出一些测试数据供参考。表中数据为单个Epoch的训练耗时，单位为秒(s)，模型名后括号内为模型训练所使用的BatchSize。

.. list-table:: 
   :widths: 25 25 25 25
   :header-rows: 1

   * - 模型
     - DataLoader
     - DataLoader+子进程
     - 加速比例
   * - Mnist (64)
     - 10.16 
     - 6.44 
     - **+68.5%**
   * - ResNet (32)
     - 83.56
     - 53.75 
     - **+75.6%**
   * - SeResNeXt (64)
     - 131.20
     - 124.49 
     - **+42.4%**
   * - Ptb (20)
     - 108.56
     - 108.27 
     - **+11.1%**
   * - MobileNet V1 (256)
     - 5041.77
     - 3249.51 
     - **+55.2%**

.. note::
    动态图DataLoader多进程方式采用共享内存机制实现Tensor的进程间传输，使用时需要保证机器上或Docker共享内存空间足够大，需要大于 :code:`DataLoader.capacity * 一个Batch的数据大小`。在物理机或者虚拟机上训练一般不会有问题，但在docker中可能出现共享内存不足的情况，因为docker中/dev/shm目录空间不够大（默认64M），若内存空间不足，建议配置较大的共享内存空间，或者仍使用单进程。