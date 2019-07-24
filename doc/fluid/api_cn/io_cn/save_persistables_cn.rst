.. _cn_api_fluid_io_save_persistables:

save_persistables
-------------------------------

.. py:function:: paddle.fluid.io.save_persistables(executor, dirname, main_program=None, filename=None)

该函数从给定 ``main_program`` 中取出所有 ``persistable==True`` 的变量，然后将它们保存到目录 ``dirname`` 中或 ``filename`` 指定的文件中。

``dirname`` 用于指定保存长期变量的目录。如果想将变量保存到指定目录的若干文件中，设置 ``filename=None`` ; 如果想将所有变量保存在一个文件中，请使用 ``filename`` 来指定它。

参数:
 - **executor**  (Executor) – 保存变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要保存变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **predicate**  (function|None) – 如果不等于None，当指定main_program， 那么只有 predicate(variable)==True 时，main_program中的变量
 - **vars**  (list[Variable]|None) –  要保存的所有变量的列表。 优先级高于main_program。默认值: None
 - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None
 
返回: None
  
**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    # `prog` 可以是由用户自定义的program
    fluid.io.save_persistables(executor=exe, dirname=param_path,
                               main_program=prog)
    
    






