.. _cn_api_profiler_profiler:

Profiler
---------------------

.. py:class:: paddle.profiler.Profiler(*, targets: Optional[Iterable[ProfilerTarget]]=None, scheduler: Union[Callable[[int], ProfilerState], tuple, None]=None, on_trace_ready: Optional[Callable[..., Any]]=None)

性能分析器，该类负责管理性能分析的启动、关闭，以及性能数据的导出和统计分析。

参数:
    - **targets** (list, 可选) - 指定性能分析所要分析的设备，默认会自动分析所有存在且支持的设备，当前为CPU和GPU（可选值见 :ref:`ProfilerState <cn_api_profiler_profilertarget>` )。
    - **scheduler** (Callable|tuple, 可选) - 性能分析器状态的调度器，默认的调度器为始终让性能分析器处于RECORD状态(详情见 :ref:`状态说明 <cn_api_profiler_profilerstate>` ）。可以自己定义调度器函数，调度器的输入是一个整数，表示当前的step, 返回值对应的性能分析器状态，可以通过 :ref:`make_scheduler <cn_api_profiler_make_scheduler>` 接口生成调度器，或者直接放个tuple, 如(2, 5),代表第2-4(不包括5，二元组表示的区间前闭后开）个step处于RECORD状态。
    - **on_trace_ready** (Callable, 可选) - 处理性能分析器的回调函数，当性能分析器处于RECORD_AND_RETURN状态或者结束时返回性能数据，将会调用on_trace_ready这个回调函数进行处理，默认为 :ref:`export_chrome_tracing <cn_api_profiler_export_chrome_tracing>` (./profiler_log/)。


.. py:method:: start()

性能分析器状态从CLOSED -> scheduler(0), 并根据新的状态触发相应行为。

返回：None。

**代码示例**

第[5-9)个step收集性能数据，并导出chrometracing文件，打印表单。

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=0.0003, parameters = linear.parameters())
    prof = profiler.Profiler(targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
                        scheduler=(5, 9),
                        on_trace_ready=profiler.export_chrome_tracing('./profiler_demo'))
    prof.start()
    for i in range(10):
        data = paddle.randn(shape=[25])
        data = paddle.reshape(data, [2, 13])
        out = linear(data)
        out.backward()
        momentum.step()
        momentum.clear_grad()
        prof.step()
    prof.stop()
    prof.summary()



.. py::method:: stop()

性能分析器状态从当前状态 -> CLOSED，性能分析器关闭，如果有性能数据返回，调用on_trace_ready回调函数进行处理。

返回：None。

**代码示例**

第[1-5)个step收集性能数据，并导出chrometracing文件，打印表单。

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=0.0003, parameters = linear.parameters())
    prof = profiler.Profiler(targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
                        scheduler=(1, 5),
                        on_trace_ready=profiler.export_chrome_tracing('./profiler_demo'))
    prof.start()
    for i in range(10):
        data = paddle.randn(shape=[25])
        data = paddle.reshape(data, [2, 13])
        out = linear(data)
        out.backward()
        momentum.step()
        momentum.clear_grad()
        prof.step()
    prof.stop()
    prof.summary()


.. py::method:: step()

指示性能分析器进入下一个step，根据scheduler计算新的性能分析器状态，并根据新的状态触发相应行为。如果有性能数据返回，调用on_trace_ready回调函数进行处理。

返回：None。

**代码示例**

收集整个执行过程的性能数据，并导出chrometracing文件，打印表单。

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=0.0003, parameters = linear.parameters())
    prof = profiler.Profiler(targets=[ProfilerTarget.CPU, ProfilerTarget.GPU],
                        on_trace_ready=profiler.export_chrome_tracing('./profiler_demo'))
    prof.start()
    for i in range(10):
        data = paddle.randn(shape=[25])
        data = paddle.reshape(data, [2, 13])
        out = linear(data)
        out.backward()
        momentum.step()
        momentum.clear_grad()
        prof.step()
    prof.stop()
    prof.summary()


.. py::method:: export(path, format="json")

导出性能数据到文件。

参数：
    - **path** (str) – 性能数据导出的文件名。
    - **format** (str, 可选) – 性能数据导出的格式，目前支持"json"和"pb"两种。即"json"为导出chrome tracing文件，"pb"为导出protobuf文件。

**代码示例**

第[5-9)个step收集性能数据，并导出protobuf文件，打印表单。

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=0.0003, parameters = linear.parameters())
    with profiler.Profiler(targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
                        scheduler=(5, 9)) as prof:
        for i in range(10):
            data = paddle.randn(shape=[25])
            data = paddle.reshape(data, [2, 13])
            out = linear(data)
            out.backward()
            momentum.step()
            momentum.clear_grad()
            prof.step()
    prof.export("profiler_data.pb", format="pb")
    prof.summary(sorted_by=SortedKeys.CPUTotal, op_detail=True, thread_sep=False, time_unit='ms')



.. py::method:: summary(sorted_by=SortedKeys.CPUTotal, op_detail=True, thread_sep=False, time_unit='ms')

统计性能数据并打印表单。当前支持从总览、模型、分布式、算子、内存操作、自定义六个角度来对性能数据进行统计。

参数：
    - **sorted_by** ( :ref:`SortedKeys <cn_api_profiler_sortedkeys>` , 可选) – 表单的数据项排列方式。
    - **op_detail** (bool, 可选) – 是否打印算子内各过程的详细信息。
    - **thread_sep** (bool, 可选) - 是否分线程打印。
    - **time_unit** (str, 可选) - 表单数据的时间单位，默认为'ms', 可选's', 'us', 'ns'。 


**代码示例**


第0个step处于CLOSED， 第[1 - 2]个step处于READY, 第[3 - 5]个step处于RECORD，在第5个step返回收集的性能数据，并导出chrome tracing文件，打印表单。

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=0.0003, parameters = linear.parameters())
    with profiler.Profiler(targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
                        scheduler=profiler.make_scheduler(closed=1, ready=2, record=3, repeat=1),
                        on_trace_ready=profiler.export_chrome_tracing('./profiler_demo')) as prof:
        for i in range(10):
            data = paddle.randn(shape=[25])
            data = paddle.reshape(data, [2, 13])
            out = linear(data)
            out.backward()
            momentum.step()
            momentum.clear_grad()
            prof.step()
    prof.summary(sorted_by=SortedKeys.CPUTotal, op_detail=True, thread_sep=False, time_unit='ms')
