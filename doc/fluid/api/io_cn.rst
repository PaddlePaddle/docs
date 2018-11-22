.. _cn_api_fluid_io_save_vars:

save_vars
>>>>>>>>>>>>

.. py:class:: paddle.fluid.io.save_vars(executor, dirname, main_program=None, vars=None, predicate=None, filename=None)

executor将变量保存到指定目录


有两种方法来保存的变量:方法一，在列表中列出变量并将其分配给vars。方法二，将已存在的 ``Program`` 赋值给 ``main_program`` ，然后将保存 ``Program`` 中的所有变量。第一种方法优先级更高。如果分配了 vars，那么忽略 ``main_program`` 和 ``predicate`` 。


``dirname`` 用于指定保存变量的目录。如果想将变量保存到指定目录的若干文件中，设置文件名 None; 如果想将所有变量保存在一个文件中，请使用filename来指定它。


参数:
 - **executor**  (Executor) – 保存变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要保存变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **vars**  (list[Variable]|None) –  要保存的所有变量的列表。 优先级高于main_program。默认值: None
 - **predicate**  (function|None) – 如果不等于None，当指定main_program， 那么只有 predicate(variable)==True 时，main_program中的变量会被保存。
 - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. Default: None
 
 
返回：None

抛出异常：
  - **TypeError** - 如果参数 ``main_program`` 为 None 或为一个非 ``Program`` 的实例
  
**代码示例**

..  code-block:: python
    
    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"

    # 第一种使用方式 使用 main_program 指定变量
    def name_has_fc(var):
        res = "fc" in var.name
        return res

    prog = fluid.default_main_program()
    fluid.io.save_vars(executor=exe, dirname=path, main_program=prog,
                       vars=None)
    # 保存“main_program”中所有名称包含“fc”的变量。
    # 分别保存变量

    # 第二种使用方式 : 使用 `vars` 指定要保存的 variables
    var_list = [var_a, var_b, var_c]
    fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                       filename="vars_file")

    # 将 var_a、var_b和var_c 分别保存在路径"为/my_paddle_model"中名为" var_a var_b 
    # var_c 的文件中
  
  
.. _cn_api_fluid_regularizer_save_params:

save_params
>>>>>>>>>>>>

.. py:class:: paddle.fluid.io.save_params(executor, dirname, main_program=None, filename=None)

该函数过滤掉 给定 ``main_program`` 中所有参数，然后将它们保存到目录 ``dirname`` 中或文件中。

``dirname`` 用于指定保存变量的目录。如果想将变量保存到指定目录的若干文件中，设置文件名 None; 如果想将所有变量保存在一个文件中，请使用filename来指定它

注意:有些变量不是参数，但它们对于训练是必要的。因此，您不能仅通过 ``save_params()`` 和 ``load_params()`` 保存并之后继续训练。可以使用 ``save_persistables()`` 和 ``load_persistables()`` 代替这两个函数


参数:
 - **executor**  (Executor) – 保存变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要保存变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **vars**  (list[Variable]|None) –  要保存的所有变量的列表。 优先级高于main_program。默认值: None
 - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None
 
返回: None
  
**代码示例**

..  code-block:: python
    
    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.save_params(executor=exe, dirname=param_path,
                         main_program=None)
