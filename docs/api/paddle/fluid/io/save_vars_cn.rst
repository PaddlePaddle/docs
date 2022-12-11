.. _cn_api_fluid_io_save_vars:

save_vars
-------------------------------


.. py:function:: paddle.fluid.io.save_vars(executor, dirname, main_program=None, vars=None, predicate=None, filename=None)




该接口将 ``Program`` 的变量保存到文件中。

通过 ``vars`` 指定需要保存的变量，或者通过 ``predicate`` 筛选需要保存的变量，``vars`` 和 ``predicate`` 不能同时为None。

参数
::::::::::::

      - **executor** （Executor）- 运行的执行器，执行器的介绍请参考 :ref:`api_guide_model_save_reader` 。
      - **dirname** （str）- 保存变量的目录路径。
      - **main_program** （Program，可选）- 需要保存变量的 ``Program`` ， ``Program`` 的介绍请参考 :ref:`api_guide_Program`。如果 ``main_program`` 为None，则使用默认的主程序。默认值为None。
      - **vars** （list [Variable]，可选）- 通过该列表指定需要保存的变量。默认值为None。
      - **predicate** （function，可选）- 通过该函数筛选 :math:`predicate(variable)== True` 的变量进行保存。如果通过 ``vars`` 指定了需要保存的变量，则该参数无效。默认值为None。
      - **filename** （str，可选）- 保存所有变量的文件。如果设置为None，所有变量会按照变量名称单独保存成文件；如果设置为非None，所有变量会保存成一个文件名为该设置值的文件。默认值为None。

返回
::::::::::::
无    

抛出异常
::::::::::::

    - ``TypeError`` - 如果main_program不是Program的实例，也不是None。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.io.save_vars