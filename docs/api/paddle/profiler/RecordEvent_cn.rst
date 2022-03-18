.. _cn_api_profiler_record_event:

RecordEvent
---------------------

.. py:class:: paddle.profiler.RecordEvent(name)

该接口用于用户自定义打点，记录某一段代码运行的时间。

参数:
    - **name** (str) - 记录打点的名字。

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

1. 调用记录函数

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    record_event = profiler.RecordEvent("record_add")
    record_event.begin()
    data1 = paddle.randn(shape=[3])
    data2 = paddle.randn(shape=[3])
    result = data1 + data2
    record_event.end()

2. 使用环境管理器的用法，with语句。

.. code-block:: python

    import paddle
    import paddle.profiler as profiler

    with profiler.RecordEvent("record_add"):
        data1 = paddle.randn(shape=[3])
        data2 = paddle.randn(shape=[3])
        result = data1 + data2
