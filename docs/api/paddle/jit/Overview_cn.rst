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
    " :ref:`ignore_module <cn_api_paddle_jit_ignore_module>` ", "增加动转静过程中忽略转写的模块"
    " :ref:`TranslatedLayer <cn_api_paddle_jit_TranslatedLayer>` ", "是一个命令式编程模式 :ref:`cn_api_paddle_nn_Layer` 的继承类"
    " :ref:`enable_to_static <cn_api_paddle_jit_enable_to_static>` ", "开启模型动转静功能接口"


.. _about_debug:

Debug 动态图转静态图相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`set_code_level <cn_api_paddle_jit_set_code_level>` ", "设置代码级别，打印该级别动转静转化后的代码"
    " :ref:`set_verbosity <cn_api_paddle_jit_set_verbosity>` ", "设置动态图转静态图的日志详细级别"
