.. _cn_overview_profiler:

paddle.profiler
---------------------

paddle.profiler 目录下包含飞桨框架的性能分析器，提供对模型训练和推理过程的
性能数据进行展示和统计分析的功能，帮助用户定位模型的性能瓶颈点。所提供的 API 具体如下:

-  :ref:`Profiler 功能使用相关的枚举类 API <about_profiler_enum>`
-  :ref:`Profiler 周期控制和性能数据 IO API <about_profiler_control>`
-  :ref:`Profiler 性能分析器 API <about_profiler_profiler>`
-  :ref:`Profiler 性能数据自定义记录 API <about_profiler_record>`



.. _about_profiler_enum:

Profiler 功能使用相关的枚举类 API
::::::::::::::::::::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`ProfilerTarget <cn_api_paddle_profiler_ProfilerTarget>` ", "用来指定性能分析的设备"
    " :ref:`ProfilerState <cn_api_paddle_profiler_ProfilerState>` ", "用来表示性能分析器的状态"
    " :ref:`SortedKeys <cn_api_paddle_profiler_SortedKeys>` ", "用来指定表单内数据的排序方式"
    " :ref:`SummaryView <cn_api_paddle_profiler_ProfilerSummaryView>` ", "用来指定数据表单类型"

.. _about_profiler_control:

Profiler 周期控制和性能数据 IO API
:::::::::::::::::::::::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`make_scheduler <cn_api_paddle_profiler_make_scheduler>` ", "用于生成性能分析器状态的调度器"
    " :ref:`export_chrome_tracing <cn_api_paddle_profiler_export_chrome_tracing>` ", "用于生成将性能数据保存到 google chrome tracing 文件的回调函数"
    " :ref:`export_protobuf <cn_api_paddle_profiler_export_protobuf>` ", "用于生成将性能数据保存到 protobuf 文件的回调函数"
    " :ref:`load_profiler_result <cn_api_paddle_profiler_load_profiler_result>` ", "用于载入所保存到 protobuf 文件的性能数据"

.. _about_profiler_profiler:

Profiler 性能分析器 API
:::::::::::::::::::::::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`Profiler <cn_api_paddle_profiler_Profiler>` ", "性能分析器"

.. _about_profiler_record:

Profiler 性能数据自定义记录 API
:::::::::::::::::::::::::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`RecordEvent <cn_api_paddle_profiler_RecordEvent>` ", "用于用户自定义打点记录时间"
