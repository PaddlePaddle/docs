.. _cn_api_profiler_record_event:

RecordEvent
---------------------

.. py:class:: paddle.profiler.RecordEvent(name)

该接口用于用户自定义打点，记录某一段代码运行的时间。


参数:
    - **name** (str) - 记录打点的名字。

**代码示例**

.. code-block:: python

    import paddle
    import paddle.profiler as profiler
    # method1: using context manager
    with profiler.RecordEvent("record_add"):
        data1 = paddle.randn(shape=[3])
        data2 = paddle.randn(shape=[3])
        result = data1 + data2
    # method2: call begin() and end()
    record_event = profiler.RecordEvent("record_add")
    record_event.begin()
    data1 = paddle.randn(shape=[3])
    data2 = paddle.randn(shape=[3])
    result = data1 + data2
    record_event.end()

注意:
    RecordEvent只有在 :ref:`性能分析器 <cn_api_profiler_profiler>` 处于RECORD状态才会生效。

.. py:method:: begin()

记录开始的时间。

**代码示例**

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    record_event = profiler.RecordEvent("record_sub")
    record_event.begin()
    data1 = paddle.randn(shape=[3])
    data2 = paddle.randn(shape=[3])
    result = data1 - data2
    record_event.end()


.. py:method:: end()

记录结束的时间。

**代码示例**

.. code-block:: python

    import paddle
    import paddle.profiler as profiler
    record_event = profiler.RecordEvent("record_mul")
    record_event.begin()
    data1 = paddle.randn(shape=[3])
    data2 = paddle.randn(shape=[3])
    result = data1 * data2
    record_event.end()
