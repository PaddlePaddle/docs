.. _cn_api_paddle_utils_profiler_profiler:

profiler
-------------------------------

.. py:function:: paddle.utils.profiler.profiler(state, sorted_key=None, profile_path='/tmp/profile', tracer_option='Default')

**profiler**接口。与 **fluid.profiler.cuda_profiler**不同，此 **profiler**可用于分析 CPU 和 GPU 程序。

参数
::::::::::::

  - **state** (str) - 分析状态，应为 ``CPU`` ， ``GPU`` 或 ``All`` 之一。 ``CPU`` 表示仅分析CPU;
    ``GPU``意味着同时分析CPU和GPU; ``All`` 意味着同时分析CPU和GPU，并生成时间线。
  - **sorted_key** (str，可选) - 分析结果的顺序，应为 ``None``、 ``calls``、 ``total``、 ``max``、 ``min``或 ``ave``之一。
    默认值为 ``None``，表示性能分析结果将按事件的第一个结束时间的顺序打印。 ``calls``意味着按调用数排序。
    ``total``意味着按总执行时间排序。 ``max``表示按最大执行时间排序。 ``min`` 表示按最小执行时间排序。 ``ave``表示按平均执行时间排序。
  - **profile_path** (str，可选) - 如果 state == 'All'，它将生成时间线，并将其写入 ``profile_path``。profile_path默认值是“/tmp/profile”。
  - **tracer_option** (str，可选) - tracer_option可以是['Default'，'OpDetail'，'AllOpDetail']之一，它可以控制配置文件级别并打印不同级别的配置文件结果。
    ``Default``选项打印不同的操作类型分析结果，OpDetail选项打印不同操作类型（如计算和数据转换）的详细性能分析结果，AllOpDetail选项打印与OpDetail相同的不同操作名称的详细性能分析结果。

异常
::::::::::::
    **ValueError** - 如果 ``state``不在 [‘CPU’, ‘GPU’, ‘All’] 中。如果sorted_key不在  [‘calls’, ‘total’, ‘max’, ‘min’, ‘ave’] 中。

代码示例
::::::::::

COPY-FROM: paddle.utils.profiler.profiler
