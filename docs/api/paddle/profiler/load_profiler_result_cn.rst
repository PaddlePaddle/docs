.. _cn_api_profiler_load_profiler_result:

load_profiler_result
-------------------------------

.. py:function:: paddle.profiler.load_profiler_result(file_name)

该接口用于载入所保存到protobuf文件的性能数据。

参数:
    - **file_name** (str) - pb格式的性能数据文件路径。

返回: ProfilerResult对象，底层存储性能数据的结构。
