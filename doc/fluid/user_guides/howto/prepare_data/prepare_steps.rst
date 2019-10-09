..  _user_guide_prepare_steps:

########
准备步骤
########

使用PaddlePaddle Fluid准备数据分为三个步骤：

Step 1: 自定义Reader生成训练/预测数据
###################################

生成的数据类型可以为Numpy Array或LoDTensor。根据Reader返回的数据形式的不同，可分为Batch级的Reader和Sample（样本）级的Reader。

Batch级的Reader每次返回一个Batch的数据，Sample级的Reader每次返回单个样本的数据

如果您的数据是Sample级的数据，我们提供了一个可以数据预处理和组建batch的工具：:code:`Python Reader` 。


Step 2: 在网络配置中定义数据层变量
###################################
用户需使用 :code:`fluid.data` 在网络中定义数据层变量。定义数据层变量时需指明数据层的名称name、数据类型dtype和维度shape。例如：

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.data(name='image', dtype='float32', shape=[None, 28, 28])
    label = fluid.data(name='label', dtype='int64', shape=[None, 1])

其中，None表示不确定的维度。此例子中None的含义为batch size。

Step 3: 将数据送入网络进行训练/预测
###################################

Fluid提供两种方式，分别是异步DataLoader接口方式或同步Feed方式，具体介绍如下：

- 异步DataLoader接口方式

用户需要先使用 :code:`fluid.io.DataLoader` 定义DataLoader对象，然后通过DataLoader对象的set方法设置数据源。
使用DataLoader接口时，数据传入与模型训练/预测过程是异步进行的，效率较高，推荐使用。

- 同步Feed方式

用户自行构造输入数据，并在 :code:`fluid.Executor` 或 :code:`fluid.ParallelExecutor`
中使用 :code:`executor.run(feed=...)` 传入训练数据。数据准备和模型训练/预测的过程是同步进行的，
效率较低。


这两种准备数据方法的比较如下:

========  =================================   =====================================
对比项            同步Feed方式                          异步DataLoader接口方式
========  =================================   =====================================
API接口     :code:`executor.run(feed=...)`          :code:`fluid.io.DataLoader`
数据格式         Numpy Array或LoDTensor               Numpy Array或LoDTensor
数据增强          Python端使用其他库完成                  Python端使用其他库完成
速度                     慢                                   快
推荐用途                调试模型                              工业训练
========  =================================   =====================================

Reader数据类型对使用方式的影响
###########################

根据Reader数据类型的不同，上述步骤的具体操作将有所不同，具体介绍如下:

读取Sample级Reader数据
+++++++++++++++++++++

若自定义的Reader每次返回单个样本的数据，用户需通过以下步骤完成数据送入：

Step 1. 组建数据
================

调用Fluid提供的Reader相关接口完成组batch和部分的数据预处理功能，具体请参见： `数据预处理工具 <./reader_cn.html>`_ 。

Step 2. 送入数据
================

若使用异步DataLoader接口方式送入数据，请调用 :code:`set_sample_generator` 或 :code:`set_sample_list_generator` 接口完成，具体请参见： :ref:`user_guides_use_py_reader` 。

若使用同步Feed方式送入数据，请使用DataFeeder接口将Reader数据转换为LoDTensor格式后送入网络，具体请参见 :ref:`cn_api_fluid_DataFeeder` 。

读取Batch级Reader数据
++++++++++++++++++++

Step 1. 组建数据
================

由于Batch已经组好，已经满足了Step 1的条件，可以直接进行Step 2。

Step 2. 送入数据
================

若使用异步DataLoader接口方式送入数据，请调用DataLoader的 :code:`set_batch_generator` 接口完成，具体方式请参见: :ref:`user_guides_use_py_reader` 。

若使用同步Feed方式送入数据，具体请参见: :ref:`user_guide_use_numpy_array_as_train_data` 。




