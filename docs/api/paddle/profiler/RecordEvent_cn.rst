.. _cn_api_profiler_record_event:

RecordEvent
---------------------

.. py:class:: paddle.profiler.RecordEvent(name: str, event_type: TracerEventType=TracerEventType.UserDefined)

用于用户自定义打点，记录某一段代码运行的时间。


参数
:::::::::

    - **name** (str) - 记录打点的名字。
    - **event_type** (TracerEventType，可选) - 可选参数，默认值为 TracerEventType.UserDefined。该参数预留为内部使用，最好不要指定该参数。

代码示例
::::::::::

COPY-FROM: paddle.profiler.RecordEvent:code-example1

.. note::
    RecordEvent 只有在 :ref:`性能分析器 <cn_api_profiler_profiler>` 处于 RECORD 状态才会生效。

方法
::::::::::::
begin()
'''''''''

记录开始的时间。

**代码示例**

COPY-FROM: paddle.profiler.RecordEvent.begin:code-example2


end()
'''''''''

记录结束的时间。

**代码示例**

COPY-FROM: paddle.profiler.RecordEvent.end:code-example3
