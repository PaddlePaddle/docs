.. _cn_overview_jit:

paddle.jit
--------------

paddle.jit 目录下包含飞桨框架支持动态图转静态图相关的 API。具体如下：

-  :ref:`动态图转静态图相关 API <about_dygraph_to_static>`
-  :ref:`Debug 动态图转静态图相关 <about_debug>`



.. _about_dygraph_to_static:

动态图转静态图相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`to_static <cn_api_paddle_jit_to_static>` ", "动转静 to_static 装饰器"
    " :ref:`save <cn_api_paddle_jit_save>` ", "动转静模型存储接口"
    " :ref:`load <cn_api_paddle_jit_load>` ", "动转静模型载入接口"
    " :ref:`ProgramTranslator <cn_api_fluid_dygraph_ProgramTranslator>` ", "动转静控制主类 ProgramTranslator"
    " :ref:`TracedLayer <cn_api_fluid_dygraph_TracedLayer>` ", "备选根据 trace 动转静的接口 TracedLayer"
    " :ref:`TranslatedLayer <cn_api_fluid_dygraph_TranslatedLayer>` ", "是一个命令式编程模式 :ref:`cn_api_fluid_dygraph_Layer` 的继承类"


.. _about_debug:

Debug 动态图转静态图相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`set_code_level <cn_api_fluid_dygraph_jit_set_code_level>` ", "设置代码级别，打印该级别动转静转化后的代码"
    " :ref:`set_verbosity <cn_api_fluid_dygraph_jit_set_verbosity>` ", "设置动态图转静态图的日志详细级别"
