.. _cn_api_profiler_load_profiler_result:

load_profiler_result
-------------------------------

.. py:function:: paddle.profiler.load_profiler_result(file_name)

该接口用于载入所保存到protobuf文件的性能数据到内存。

参数
:::::::::

    - **file_name** (str) - protobuf格式的性能数据文件路径。

返回
:::::::::

ProfilerResult对象，底层存储性能数据的结构。

代码示例
::::::::::

COPY-FROM: paddle.profiler.load_profiler_result:code-example1
