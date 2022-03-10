.. _cn_overview_profiler:

paddle.profiler
---------------------

paddle.profiler 目录下包含飞桨框架的性能分析器，提供对模型训练和推理过程的
性能数据进行展示和统计分析的功能，帮助用户定位模型的性能瓶颈点。所提供的API具体如下:

.. py:class:: ProfilerTarget
    .. py:attribute:: CPU
    .. py:attribute:: GPU

枚举类，用来指定性能分析的设备。目前仅支持CPU和GPU。

.. py:class:: ProfilerState
    .. py::attribute:: CLOSED
    .. py::attribute:: READY
    .. py::attribute:: RECORD
    .. py::attribute:: RECORD_AND_RETURN

枚举类，用来表示性能分析器的状态。

状态说明如下：
    - **CLOSED** - "关闭"，不收集任何性能数据。
    - **READY**  - "准备"，性能分析器开启，但是不做数据记录，该状态主要为了减少性能分析器初始化时的开销对所收集的性能数据的影响。
    - **RECORD** - "记录"，性能分析器正常工作，并且记录性能数据。
    - **RECORD_AND_RETURN** - "记录并返回"，性能分析器正常工作，并且将记录的性能数据返回。

.. py:function:: make_scheduler(*, closed: int, ready: int, record: int, repeat: int=0, skip_first: int=0)

该接口用于生成性能分析器状态的调度器。

参数:
    - **closed** (int) - 处于ProfilerState.CLOSED状态的step数量。
    - **ready** (int) - 处于ProfilerState.CLOSED状态的step数量。
    - **record** (int) - 处于ProfilerState.RECORD状态的step数量, record的最后一个step会处于ProfilerState.RECORD_AND_RETURN状态。
    - **repeat** (int, 可选) - 调度器重复该状态调度过程的次数, 默认值为0意味着一直重复该调度过程。
    - **skip_first** (int, 可选) - 跳过前skip_first个step，不参与状态调度，并处于ProfilerState.CLOSED状态。

返回: 调度函数（callable), 该函数会接收一个参数step_num，并计算返回相应的ProfilerState。性能分析器可以通过该调度函数更新状态。


**代码示例**
1. 第0个step处于CLOSED， 第[1 - 2]个step处于READY, 第[3 - 5]个step处于RECORD，在第5个step返回收集的性能数据。
.. code-block:: python

    import paddle.profiler as profiler
    sheduler = profiler.make_scheduler(closed=1, ready=2, record=3, repeat=1)

2. 第0个step处于CLOSED， 第[1 - 3]个step处于READY, 第[4 - 6]个step处于RECORD，在第5个step返回收集的性能数据，重复3次。即第7个step处于CLOSED，第[8-10]个step处于READY,
第[11-13]个step处于RECORD，并在第13个step返回第二轮所收集到的性能数据。以此类推，直到第20个step返回第三轮所收集到的性能数据，调度结束。
.. code-block:: python

    import paddle.profiler as profiler
    sheduler = profiler.make_scheduler(closed=1, ready=3, record=3, repeat=3)


.. py:function:: export_chrome_tracing(dir_name: str, worker_name: Optional[str]=None)

该接口用于生成将性能数据保存到google chrome tracing文件的回调函数。

参数:
    - **dir_name** (str) - 性能数据导出所保存到的文件夹路径。
    - **worker_name** (str, 可选) - 性能数据导出所保存到的文件名前缀，默认是[hostname]_[pid]。

返回: 回调函数（callable), 该函数会接收一个参数prof(Profiler对象），调用prof的export方法保存采集到的性能数据到chrome tracing文件。


.. py:function:: export_protobuf(dir_name: str, worker_name: Optional[str]=None)

该接口用于生成将性能数据保存到protobuf文件的回调函数。

参数:
    - **dir_name** (str) - 性能数据导出所保存到的文件夹路径。
    - **worker_name** (str, 可选) - 性能数据导出所保存到的文件名前缀，默认是[hostname]_[pid]。

返回: 回调函数（callable), 该函数会接收一个参数prof(Profiler对象），调用prof的export方法保存采集到的性能数据到protobuf文件。


.. py:class:: Profiler(*, targets: Optional[Iterable[ProfilerTarget]]=None, scheduler: Union[Callable[[int], ProfilerState], tuple, None]=None, on_trace_ready: Optional[Callable[..., Any]]=None)

性能分析器，该类负责管理性能分析的启动、关闭，以及性能数据的导出和统计分析。

参数:
    - **targets** (list, 可选) - 指定性能分析所要分析的设备，默认会自动分析所有存在且支持的设备，当前为CPU和GPU。
    - **scheduler** (Callable|tuple, 可选) - 性能分析器状态的调度器，默认的调度器为始终让性能分析器处于RECORD状态。可以通过make_scheduler接口生成调度器，或者直接放个tuple, 如(2, 5),代表第2-4(不包括5，二元组表示的区间前闭后开）个step处于RECORD状态。
    - **on_trace_ready** (Callable, 可选) - 处理性能分析器的回调函数，当性能分析器处于RECORD_AND_RETURN状态或者结束时返回性能数据，将会调用on_trace_ready这个回调函数进行处理，默认为export_chrome_tracing(./profiler_log/)。

.. py:method:: start()

性能分析器状态从CLOSED -> scheduler(0), 并根据新的状态触发相应行为。

.. py::method:: stop()

性能分析器状态从当前状态 -> CLOSED，性能分析器关闭，如果有性能数据返回，调用on_trace_ready回调函数进行处理。

.. py::method:: step()

指示性能分析器进入下一个step，根据scheduler计算新的性能分析器状态，并根据新的状态触发相应行为。如果有性能数据返回，调用on_trace_ready回调函数进行处理。

.. py::method:: export(path, format="json"):

导出性能数据到文件。

参数：
    - **path** (str) – 性能数据导出的文件名。
    - **format** (str, 可选) – 性能数据导出的格式，目前支持"json"和"pb"两种。即"json"为导出chrome tracing文件，"pb"为导出protobuf文件。

.. py::method:: summary(sorted_by=SortedKeys.CPUTotal, op_detail=True, thread_sep=False, time_unit='ms')

统计性能数据并打印表单。当前支持从总览、模型、分布式、算子、内存操作、自定义六个角度来对性能数据进行统计。

参数：
    - **sorted_by** (SortedKeys, 可选) – 表单的数据项排列方式。
    - **op_detail** (bool, 可选) – 是否打印算子内各过程的详细信息。
    - **thread_sep** (bool, 可选) - 是否分线程打印。
    - **time_unit** (str, 可选) - 表单数据的时间单位，默认为'ms', 可选's', 'us', 'ns'。 



**代码示例**

1. 第0个step处于CLOSED， 第[1 - 2]个step处于READY, 第[3 - 5]个step处于RECORD，在第5个step返回收集的性能数据，并导出chrome tracing文件，打印表单。

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
    prof.summary()

2. 第[5-9)个step收集性能数据，并导出chrometracing文件，打印表单。

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


.. py:class:: RecordEvent(name)

该接口用于用户自定义打点，记录某一段代码运行的时间。

参数:
    - **name** (str) - 记录打点的名字。

.. py:method:: begin()

记录开始的时间。

.. py:method:: end()

记录结束的时间。

**代码示例**

1. 使用环境管理器的用法，with语句。

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    with profiler.RecordEvent("record_add"):
      data1 = paddle.randn(shape=[3])
      data2 = paddle.randn(shape=[3])
      result = data1 + data2

2. 手动调用记录函数

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    record_event = profiler.RecordEvent("record_add")
    record_event.begin()
    data1 = paddle.randn(shape=[3])
    data2 = paddle.randn(shape=[3])
    result = data1 + data2
    record_event.end()


.. py:class:: SortedKeys
  .. py::attribute:: CPUTotal
  .. py::attribute:: CPUAvg
  .. py::attribute:: CPUMax
  .. py::attribute:: CPUMin
  .. py::attribute:: GPUTotal
  .. py::attribute:: GPUAvg
  .. py::attribute:: GPUMax
  .. py::attribute:: GPUMin

枚举类，用来指定表单内数据的排序方式。


.. py:function:: load_profiler_result(file_name)

该接口用于载入所保存到protobuf文件的性能数据。

参数:
    - **file_name** (str) - pb格式的性能数据文件路径。

返回: 结构化的性能数据。

