.. _cn_api_fluid_profiler_stop_profiler:

stop_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.stop_profiler(sorted_key=None, profile_path='/tmp/profile')

停止 profiler， 用户可以使用 ``fluid.profiler.start_profiler`` 和 ``fluid.profiler.stop_profiler`` 插入代码
不能使用 ``fluid.profiler.profiler``

参数:
  - **sorted_key** (string) – 如果为None，prfile的结果将按照事件的第一次结束时间顺序打印。否则，结果将按标志排序。标志取值为"call"、"total"、"max"、"min" "ave"之一，根据调用着的数量进行排序。total表示按总执行时间排序，max 表示按最大执行时间排序。min 表示按最小执行时间排序。ave表示按平均执行时间排序。
  - **profile_path** (string) - 如果 state == 'All', 结果将写入文件 profile proto.


抛出异常:
  - ``ValueError`` – 如果state 取值不在 ['CPU', 'GPU', 'All']中

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.profiler as profiler

    profiler.start_profiler('GPU')
    for iter in range(10):
        if iter == 2:
            profiler.reset_profiler()
            # except each iteration
    profiler.stop_profiler('total', '/tmp/profile')







