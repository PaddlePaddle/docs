.. _cn_api_profiler_load_profiler_result:

load_profiler_result
-------------------------------

.. py:function:: paddle.profiler.load_profiler_result(file_name)

该接口用于载入所保存到protobuf文件的性能数据。

参数:
    - **file_name** (str) - pb格式的性能数据文件路径。

返回: ProfilerResult对象，底层存储性能数据的结构。

**代码示例**

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=0.0003, parameters = linear.parameters())
    with profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            scheduler = (3, 10)) as p:
        for iter in range(10):
            data = paddle.randn(shape=[26])
            data = paddle.reshape(data, [2, 13])
            out = linear(data)
            out.backward()
            momentum.step()
            momentum.clear_grad()
            p.step()
    p.export('test_export_protobuf.pb', format='pb')
    profiler_result = profiler.load_profiler_result('test_export_protobuf.pb')
