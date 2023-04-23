..  _api_guide_data_in_out:

数据输入输出
###############


数据输入
-------------

Fluid 支持两种数据输入方式，包括：

1. Python Reader: 纯 Python 的 Reader。用户在 Python 端定义 :code:`fluid.layers.data` 层构建网络，并通过
:code:`executor.run(feed=...)` 的方式读入数据。数据读取和模型训练/预测的过程是同步进行的。

2. PyReader: 高效灵活的 C++ Reader 接口。PyReader 内部维护容量为 :code:`capacity` 的队列（队列容量由
:code:`fluid.layers.py_reader` 接口中的 :code:`capacity` 参数设置），Python 端调用队列的 :code:`push`
方法送入训练/预测数据，C++端的训练/预测程序调用队列的 :code:`pop` 方法取出 Python 端送入的数据。PyReader 可与
:code:`double_buffer` 配合使用，实现数据读取和训练/预测的异步执行。

具体使用方法请参考 :ref:`cn_api_fluid_layers_py_reader`。


数据输出
------------

Fluid 支持在训练/预测阶段获取当前 batch 的数据。

用户可通过 :code:`executor.run(fetch_list=[...], return_numpy=...)` 的方式
fetch 期望的输出变量，通过设置 :code:`return_numpy` 参数设置是否将输出数据转为 numpy array。
若 :code:`return_numpy` 为 :code:`False` ，则返回 :code:`LoDTensor` 类型数据。

具体使用方式请参考相关 API 文档 :ref:`cn_api_fluid_executor_Executor` 和
:ref:`cn_api_fluid_ParallelExecutor`。
