.. _cn_overview_jit:

paddle.jit
--------------

paddle.jit 目录下包含飞桨框架支持动态图转静态图相关的API。具体如下：

-  :ref:`动态图转静态图相关API <about_dygraph_to_static>`
-  :ref:`Debug动态图转静态图相关 <about_debug>`



.. _about_dygraph_to_static:

动态图转静态图相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`to_static <cn_api_paddle_jit_to_static>` ", "动转静to_static装饰器"
    " :ref:`save <cn_api_paddle_jit_save>` ", "动转静模型存储接口"
    " :ref:`load <cn_api_paddle_jit_load>` ", "动转静模型载入接口"
    " :ref:`ProgramTranslator <cn_api_fluid_dygraph_ProgramTranslator>` ", "动转静控制主类ProgramTranslator"
    " :ref:`TracedLayer <cn_api_fluid_dygraph_TracedLayer>` ", "备选根据trace动转静的接口TracedLayer"
    

.. _about_debug:

Debug动态图转静态图相关
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`set_code_level <cn_api_fluid_dygraph_jit_set_code_level>` ", "设置代码级别，打印该级别动转静转化后的代码"
    " :ref:`set_verbosity <cn_api_fluid_dygraph_jit_set_verbosity>` ", "设置动态图转静态图的日志详细级别"

