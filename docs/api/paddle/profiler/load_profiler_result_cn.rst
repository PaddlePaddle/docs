.. _cn_api_paddle_profiler_load_profiler_result:

load_profiler_result
-------------------------------

.. py:function:: paddle.profiler.load_profiler_result(file_name: str)

载入所保存到 protobuf 文件的性能数据到内存。

参数
:::::::::

    - **file_name** (str) - protobuf 格式的性能数据文件路径。

返回
:::::::::

ProfilerResult 对象，底层存储性能数据的结构。

代码示例
::::::::::

COPY-FROM: paddle.profiler.load_profiler_result
