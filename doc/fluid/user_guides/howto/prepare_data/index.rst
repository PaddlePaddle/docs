..  _user_guide_prepare_data:

########
准备数据
########

使用PaddlePaddle Fluid准备数据分为两个步骤：

- 自定义Reader生成训练/预测数据，生成的数据类型可以为Numpy Array或LoDTensor。根据Reader返回的数据形式的不同，可分为Batch级的Reader（每次返回一个batch的数据）和样本级的Reader（每次返回单个样本的数据）。

- 使用同步Feed方式或异步py_reader接口方式将数据送入网络进行训练/预测。

同步Feed方式和异步py_reader接口方式的比较
######################################

同步Feed方式和异步py_reader接口方式的不同体现于：

1. 同步Feed方式：用户需使用 :code:`fluid.layers.data`
配置数据输入层，并在 :code:`fluid.Executor` 或 :code:`fluid.ParallelExecutor`
中使用 :code:`executor.run(feed=...)` 传入训练数据。数据准备和模型训练/预测的过程是同步进行的，
效率较低。

2. 异步py_reader接口方式：用户需要先使用 :code:`fluid.layers.py_reader` 配置数据输入层，然后使用
:code:`py_reader` 的 :code:`decorate_paddle_reader` 或 :code:`decorate_tensor_provider`
方法配置数据源，再通过 :code:`fluid.layers.read_file` 读取数据。数据传入与模型训练/预测过程是异步进行的，
效率较高。


这两种准备数据方法的比较如下:

========  =================================   =====================================
对比项            同步Feed方式                          异步py_reader接口方式
========  =================================   =====================================
API接口     :code:`executor.run(feed=...)`       :code:`fluid.layers.py_reader`
数据格式         Numpy Array或LoDTensor               Numpy Array或LoDTensor
数据增强          Python端使用其他库完成                  Python端使用其他库完成
速度                     慢                                   快
推荐用途                调试模型                              工业训练
========  =================================   =====================================

读取Batch级Reader数据
#######################

若使用同步Feed方式送入数据，具体请参见:

.. toctree::
   :maxdepth: 1

   feeding_data.rst

若使用异步py_reader接口方式送入数据，请调用py_reader的 :code:`decorate_tensor_provider` 接口完成，具体方式请参见:

.. toctree::
   :maxdepth: 1

   use_py_reader.rst

读取样本级Reader数据
#####################

若自定义的Reader每次返回单个样本的数据，用户需通过以下步骤完成数据送入：

1.调用Fluid提供的Reader相关接口完成组batch和部分的数据预处理功能，具体请参见：

.. toctree::
   :maxdepth: 1

   reader_cn.md

2.若使用同步Feed方式送入数据，请使用DataFeeder接口将Reader数据转换为LoDTensor格式后送入网络，具体请参见 :ref:`cn_api_fluid_DataFeeder` 。若使用异步py_reader接口方式送入数据，请调用 :code:`decorate_paddle_reader` 接口完成，具体请参见：

- :ref:`user_guides_use_py_reader`
