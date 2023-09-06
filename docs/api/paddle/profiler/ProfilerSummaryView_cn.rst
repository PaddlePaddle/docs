.. _cn_api_profiler_summaryview:

SummaryView
---------------------

.. py:class:: paddle.profiler.SummaryView

SummaryView 枚举类用来表示数据表单的类型。

类型说明
::::::::::::

    - **SummaryView.DeviceView** - 设备类型数据表单。
    - **SummaryView.OverView** - 概览类型数据表单。
    - **SummaryView.ModelView** - 模型类型数据表单。
    - **SummaryView.DistributedView** - 分布式类型数据表单。
    - **SummaryView.KernelView** - 内核类型数据表单。
    - **SummaryView.OperatorView** - OP 类型数据表单。
    - **SummaryView.MemoryView** - 内存类型数据表单。
    - **SummaryView.MemoryManipulationView** - 内存操作类型数据表单。
    - **SummaryView.UDFView** - 用户自定义类型数据表单。
