.. _cn_api_profiler_load_profiler_result:

load_profiler_result
-------------------------------

.. py:function:: paddle.profiler.load_profiler_result(file_name)

该接口用于载入所保存到protobuf文件的性能数据到内存。

参数:
    - **file_name** (str) - protobuf格式的性能数据文件路径。

返回: ProfilerResult对象，底层存储性能数据的结构。

**代码示例**

.. code-block:: python

    import paddle.profiler as profiler
    with profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            scheduler = (3, 10)) as p:
        for iter in range(10):
            #train()
            p.step()
    p.export('test_export_protobuf.pb', format='pb')
    profiler_result = profiler.load_profiler_result('test_export_protobuf.pb')