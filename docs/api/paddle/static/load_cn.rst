.. _cn_api_fluid_load:

load
-------------------------------

.. py:function:: paddle.static.load(program, model_path, executor=None, var_list=None)


该接口从Program中过滤出参数和优化器信息，然后从文件中获取相应的值。

如果Program和加载的文件之间参数的维度或数据类型不匹配，将引发异常。

该函数还可以加载用[save_params，save_persistables，save_vars]接口保存的模型文件。
当[save_params，save_persistables，save_vars]保存的模型格式为单个大文件时，var_list不能为None。

参数
::::::::::::

 - **program**  ( :ref:`cn_api_fluid_Program` ) – 要加载的Program。
 - **model_path**  (str) – 保存Program的目录名称+文件前缀。格式为 ``目录名称/文件前缀`` 。
 - **executor** (Executor，可选) - 当startup program没有运行时，用于初始化参数的Executor。默认值：None。
 - **var_list** (list，可选) - 指定加载的Tensor列表，该参数只在加载旧接口[save_params，save_persistables，save_vars]保存的模型文件时使用。当加载的是多个小文件时，Tensor列表可以是所有加载文件中Tensor的子集；当加载的单个大文件时，Tensor列表必须和加载文件中的Tensor保持一致。

返回
::::::::::::
无。

代码示例
::::::::::::

COPY-FROM: paddle.static.load