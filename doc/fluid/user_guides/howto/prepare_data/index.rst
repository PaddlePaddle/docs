..  _user_guide_prepare_data:

########
准备数据
########

PaddlePaddle Fluid支持两种传入数据的方式:

1. Python Reader同步方式：用户需要使用 :code:`fluid.layers.data`
配置数据输入层，并在 :code:`fluid.Executor` 或 :code:`fluid.ParallelExecutor`
中，使用 :code:`executor.run(feed=...)` 传入训练数据。

2. py_reader接口异步方式：用户需要先使用 :code:`fluid.layers.py_reader` 配置数据输入层，然后使用
:code:`py_reader` 的 :code:`decorate_paddle_reader` 或 :code:`decorate_tensor_provider`
方法配置数据源，再通过 :code:`fluid.layers.read_file` 读取数据。


这两种准备数据方法的比较如下:

========  =================================   =====================================
对比项            Python Reader同步方式                py_reader接口异步方式
========  =================================   =====================================
API接口     :code:`executor.run(feed=...)`       :code:`fluid.layers.py_reader`
数据格式              Numpy Array                   Numpy Array或LoDTensor
数据增强          Python端使用其他库完成                  Python端使用其他库完成
速度                     慢                                   快
推荐用途                调试模型                              工业训练
========  =================================   =====================================

Python Reader同步方式
#####################

Fluid提供Python Reader方式传入数据。
Python Reader是纯的Python端接口，数据传入与模型训练/预测过程是同步的。用户可通过Numpy Array传入
数据，具体请参考:

.. toctree::
   :maxdepth: 1

   feeding_data.rst

Python Reader支持组batch、shuffle等高级功能，具体请参考：

.. toctree::
   :maxdepth: 1

   reader_cn.md

py_reader接口异步方式
#####################

Fluid提供PyReader异步数据传入方式，数据传入与模型训练/预测过程是异步的，效率较高。具体请参考：

.. toctree::
   :maxdepth: 1

   use_py_reader.rst
