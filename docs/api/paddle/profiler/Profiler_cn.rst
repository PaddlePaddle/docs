.. _cn_api_profiler_profiler:

Profiler
---------------------

.. py:class:: paddle.profiler.Profiler(*, targets: Optional[Iterable[ProfilerTarget]]=None, scheduler: Union[Callable[[int], ProfilerState], tuple, None]=None, on_trace_ready: Optional[Callable[..., Any]]=None)

性能分析器，该类负责管理性能分析的启动、关闭，以及性能数据的导出和统计分析。

参数
:::::::::

    - **targets** (list, 可选) - 指定性能分析所要分析的设备，默认会自动分析所有存在且支持的设备，当前为CPU和GPU（可选值见 :ref:`ProfilerState <cn_api_profiler_profilertarget>` )。
    - **scheduler** (Callable|tuple, 可选) - 如果是Callable对象，代表是性能分析器状态的调度器，该调度器会接受一个step_num参数并返回相应的状态(详情见 :ref:`状态说明 <cn_api_profiler_profilerstate>` ），可以通过 :ref:`make_scheduler <cn_api_profiler_make_scheduler>` 接口生成调度器。如果没有设置这个参数(None)，默认的调度器会一直让性能分析器保持RECORD状态到结束。如果是tuple类型, 有两个值start_batch和end_batch，则会在[start_batch, end_batch)(前闭后开区间)内处于RECORD状态进行性能分析。
    - **on_trace_ready** (Callable, 可选) - 处理性能分析器的回调函数，该回调函数接受Profiler对象作为参数，提供了一种自定义后处理的方式。当性能分析器处于RECORD_AND_RETURN状态或者结束时返回性能数据，将会调用该回调函数进行处理，默认为 :ref:`export_chrome_tracing <cn_api_profiler_export_chrome_tracing>` (./profiler_log/)。

代码示例
::::::::::

1. 性能分析 batch [2, 5)

COPY-FROM: paddle.profiler.Profiler:code-example1

2. 性能分析 batch [2,4], [7, 9], [11,13]

COPY-FROM: paddle.profiler.Profiler:code-example2

3. 使用全部默认参数，且脱离环境管理器的用法，性能分析整个运行过程

COPY-FROM: paddle.profiler.Profiler:code-example3

.. py:method:: start()

开启性能分析器，进入状态scheduler(0)。即
性能分析器状态从CLOSED -> scheduler(0), 并根据新的状态触发相应行为。

代码示例
::::::::::

COPY-FROM: paddle.profiler.Profiler.start:code-example4


.. py:method:: stop()

停止性能分析器，并且进入状态CLOSED。即
性能分析器状态从当前状态 -> CLOSED，性能分析器关闭，如果有性能数据返回，调用on_trace_ready回调函数进行处理。

代码示例
::::::::::

COPY-FROM: paddle.profiler.Profiler.stop:code-example5


.. py:method:: step()

指示性能分析器进入下一个step，根据scheduler计算新的性能分析器状态，并根据新的状态触发相应行为。如果有性能数据返回，调用on_trace_ready回调函数进行处理。


代码示例
::::::::::

COPY-FROM: paddle.profiler.Profiler.step:code-example6


.. py:method:: export(path, format="json")

导出性能数据到文件。

参数
:::::::::

    - **path** (str) – 性能数据导出的文件名。
    - **format** (str, 可选) – 性能数据导出的格式，目前支持"json"和"pb"两种。即"json"为导出chrome tracing文件，"pb"为导出protobuf文件，默认值为"json"。

代码示例
::::::::::

COPY-FROM: paddle.profiler.Profiler.export:code-example7


.. _cn_api_profiler_profiler_summary:

.. py:method:: summary(sorted_by=SortedKeys.CPUTotal, op_detail=True, thread_sep=False, time_unit='ms')

统计性能数据并打印表单。当前支持从总览、模型、分布式、算子、内存操作、自定义六个角度来对性能数据进行统计。

参数
:::::::::

    - **sorted_by** ( :ref:`SortedKeys <cn_api_profiler_sortedkeys>` , 可选) – 表单的数据项排列方式, 默认值SortedKeys.CPUTotal。
    - **op_detail** (bool, 可选) – 是否打印算子内各过程的详细信息，默认值True。
    - **thread_sep** (bool, 可选) - 是否分线程打印，默认值False。
    - **time_unit** (str, 可选) - 表单数据的时间单位，默认为'ms', 可选's', 'us', 'ns'。 


代码示例
::::::::::

COPY-FROM: paddle.profiler.Profiler.summary:code-example8