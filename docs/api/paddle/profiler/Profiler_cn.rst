.. _cn_api_profiler_profiler:

Profiler
---------------------

.. py:class:: paddle.profiler.Profiler(*, targets: Optional[Iterable[ProfilerTarget]]=None, scheduler: Union[Callable[[int], ProfilerState], tuple, None]=None, on_trace_ready: Optional[Callable[..., Any]]=None, record_shapes: Optional[bool] = False, profile_memory: Optional[bool] = False, timer_only: Optional[bool]=False, custom_device_types: Optional[list] = [], with_flops: Optional[bool] = False)

性能分析器，该类负责管理性能分析的启动、关闭，以及性能数据的导出和统计分析。

参数
:::::::::

    - **targets** (list，可选) - 指定性能分析所要分析的设备，默认会自动分析所有存在且支持的设备，当前支持 CPU，GPU 和 MLU（可选值见 :ref:`ProfilerState <cn_api_profiler_profilertarget>` )。
    - **scheduler** (Callable|tuple，可选) - 如果是 Callable 对象，代表是性能分析器状态的调度器，该调度器会接受一个 step_num 参数并返回相应的状态(详情见 :ref:`状态说明 <cn_api_profiler_profilerstate>` ），可以通过 :ref:`make_scheduler <cn_api_profiler_make_scheduler>` 接口生成调度器。如果没有设置这个参数(None)，默认的调度器会一直让性能分析器保持 RECORD 状态到结束。如果是 tuple 类型，有两个值 start_batch 和 end_batch，则会在[start_batch, end_batch)(前闭后开区间)内处于 RECORD 状态进行性能分析。
    - **on_trace_ready** (Callable，可选) - 处理性能分析器的回调函数，该回调函数接受 Profiler 对象作为参数，提供了一种自定义后处理的方式。当性能分析器处于 RECORD_AND_RETURN 状态或者结束时返回性能数据，将会调用该回调函数进行处理，默认为 :ref:`export_chrome_tracing <cn_api_profiler_export_chrome_tracing>` (./profiler_log/)。
    - **record_shapes** (bool，可选) - 如果设置为 True, 则会开启收集框架算子输入 Tensor 的 shape，默认为 False。
    - **profile_memory** (bool，可选) - 如果设置为 True, 则会开启收集显存分析的数据，默认为 False。
    - **timer_only** (bool，可选) - 如果设置为 True，将只统计模型的数据读取和每一个迭代所消耗的时间，而不进行性能分析。否则，模型将被计时，同时进行性能分析。默认值：False。
    - **custom_device_types** (list，可选) - 如果 targets 包含 profiler.ProfilerTarget.CUSTOM_DEVICE，custom_device_types 选择进行性能分析的设备类型，默认值为空表示所有自定义设备。
    - **with_flops** (bool，可选) - 如果设置为 True，则会开启收集相关数据计算 FLOPs 指标。默认值：False。

代码示例 1
::::::::::

性能分析 batch [2, 5)

COPY-FROM: paddle.profiler.Profiler:code-example1

代码示例 2
::::::::::

性能分析 batch [2,4], [7, 9], [11,13]

COPY-FROM: paddle.profiler.Profiler:code-example2

代码示例 3
::::::::::

使用全部默认参数，且脱离环境管理器的用法，性能分析整个运行过程

COPY-FROM: paddle.profiler.Profiler:code-example3

代码示例 4
::::::::::

使用该工具获取模型的吞吐量以及模型的时间开销

COPY-FROM: paddle.profiler.Profiler:code-example-timer1

方法
::::::::::::

start()
'''''''''

开启性能分析器，进入状态 scheduler(0)。即
性能分析器状态从 CLOSED -> scheduler(0)，并根据新的状态触发相应行为。

**代码示例**

COPY-FROM: paddle.profiler.Profiler.start:code-example4


stop()
'''''''''

停止性能分析器，并且进入状态 CLOSED。即
性能分析器状态从当前状态 -> CLOSED，性能分析器关闭，如果有性能数据返回，调用 on_trace_ready 回调函数进行处理。

**代码示例**

COPY-FROM: paddle.profiler.Profiler.stop:code-example5


step(num_samples: Optional[int]=None)
'''''''''

指示性能分析器进入下一个 step，根据 scheduler 计算新的性能分析器状态，并根据新的状态触发相应行为。如果有性能数据返回，调用 on_trace_ready 回调函数进行处理。

**参数**

    - **num_samples** (int|None，可选) - 模型运行中每一步的样本数量 batch size，当 timer_only 为 True 时该参数被用于计算吞吐量。默认值：None。

**代码示例**

COPY-FROM: paddle.profiler.Profiler.step:code-example6


step_info(unit: Optional[int]=None)
'''''''''

获取当前迭代的统计信息。如果以特定的迭代间隔调用该方法，则结果是上一次调用和本次调用之间所有迭代的平均值。统计信息如下：

1. reader_cost：加载数据的开销，单位为秒。

2. batch_cost：1 次迭代的开销，单位为秒。

3. ips（Instance Per Second）：模型吞吐量，单位为 samples/s 或其他，取决于参数 unit 的设置。当 step()的 num_samples 为 None 时，单位为 steps/s。

**参数**

    - **unit** (string，可选) - 输入数据的单位，仅在 step()的 num_samples 指定为实数时有效。例如，当 unit 为 images 时，吞吐量的单位为 images/s。默认值：None，吞吐量的单位是 samples/s。

**返回**

表示统计数据的字符串

**代码示例**

COPY-FROM: paddle.profiler.Profiler.step_info:code-example-timer2


export(path, format="json")
'''''''''

导出性能数据到文件。

**参数**

    - **path** (str) – 性能数据导出的文件名。
    - **format** (str，可选) – 性能数据导出的格式，目前支持"json"和"pb"两种。即"json"为导出 chrome tracing 文件，"pb"为导出 protobuf 文件，默认值为"json"。

**代码示例**

COPY-FROM: paddle.profiler.Profiler.export:code-example7


.. _cn_api_profiler_profiler_summary:

summary(sorted_by=SortedKeys.CPUTotal, op_detail=True, thread_sep=False, time_unit='ms', views=None)
'''''''''

统计性能数据并打印表单。当前支持从总览、模型、分布式、算子、内存操作、自定义六个角度来对性能数据进行统计。

**参数**

    - **sorted_by** ( :ref:`SortedKeys <cn_api_profiler_sortedkeys>`，可选) – 表单的数据项排列方式，默认值 SortedKeys.CPUTotal。
    - **op_detail** (bool，可选) – 是否打印算子内各过程的详细信息，默认值 True。
    - **thread_sep** (bool，可选) - 是否分线程打印，默认值 False。
    - **time_unit** (str，可选) - 表单数据的时间单位，默认为'ms'，可选's'、'us'、'ns'。
    - **views** (SummaryView|list[SummaryView]，可选) - 打印的表单列表，默认值为 None, 表示打印所有表单。


**代码示例**

COPY-FROM: paddle.profiler.Profiler.summary
