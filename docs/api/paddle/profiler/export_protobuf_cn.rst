.. _cn_api_profiler_export_protobuf:

export_protobuf
---------------------

.. py:function:: paddle.profiler.export_protobuf(dir_name: str, worker_name: Optional[str]=None)

该接口用于生成将性能数据保存到protobuf文件的回调函数。

参数:
    - **dir_name** (str) - 性能数据导出所保存到的文件夹路径。
    - **worker_name** (str, 可选) - 性能数据导出所保存到的文件名前缀，默认是[hostname]_[pid]。

返回: 回调函数（callable), 该函数会接收一个参数prof(Profiler对象），调用prof的export方法保存采集到的性能数据到protobuf文件。

**代码示例**

用于 :ref:`性能分析器 <cn_api_profiler_profiler>` 的on_trace_ready参数。

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=0.0003, parameters = linear.parameters())
    with profiler.Profiler(targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
                        scheduler=(3, 9),
                        on_trace_ready=profiler.export_protobuf('./profiler_demo')) as prof:
        for i in range(10):
            data = paddle.randn(shape=[26])
            data = paddle.reshape(data, [2, 13])
            out = linear(data)
            out.backward()
            momentum.step()
            momentum.clear_grad()
            prof.step()