.. _cn_api_fluid_io_save_vars:

save_vars
>>>>>>>>>>>>

.. py:class:: paddle.fluid.io.save_vars(executor, dirname, main_program=None, vars=None, predicate=None, filename=None)

executor将变量保存到指定目录


有两种方法来保存变量:方法一，``vars`` 为变量的列表。方法二，将已存在的 ``Program`` 赋值给 ``main_program`` ，然后将保存 ``Program`` 中的所有变量。第一种方法优先级更高。如果指定了 vars，那么忽略 ``main_program`` 和 ``predicate`` 。


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
  
  
.. _cn_api_fluid_io_save_params:

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
                         
.. _cn_api_fluid_io_save_persistables:

save_persistables
>>>>>>>>>>>>

.. py:class:: paddle.fluid.io.save_persistables(executor, dirname, main_program=None, filename=None)

该函数过滤掉 给定 ``main_program`` 中所有参数，然后将它们保存到目录 ``dirname`` 中或文件中。

``dirname`` 用于指定保存变量的目录。如果想将变量保存到指定目录的若干文件中，设置文件名 None; 如果想将所有变量保存在一个文件中，请使用filename来指定它

参数:
 - **executor**  (Executor) – 保存变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要保存变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **predicate**  (function|None) – 如果不等于None，当指定main_program， 那么只有 predicate(variable)==True 时，main_program中的变量
- **vars**  (list[Variable]|None) –  要保存的所有变量的列表。 优先级高于main_program。默认值: None
 - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None
 
返回: None
  
**代码示例**

..  code-block:: python
    
    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.save_persistables(executor=exe, dirname=param_path,
                               main_program=None)
    
    
.. _cn_api_fluid_io_load_vars:

load_vars
>>>>>>>>>>>>

.. py:class:: paddle.fluid.io.load_vars(executor, dirname, main_program=None, vars=None, predicate=None, filename=None)

``executor`` 从指定目录加载变量。

有两种方法来加载变量:方法一，``vars`` 为变量的列表。方法二，将已存在的 ``Program`` 赋值给 ``main_program`` ，然后将保存 ``Program`` 中的所有变量。第一种方法优先级更高。如果指定了 vars，那么忽略 ``main_program`` 和 ``predicate`` 。

``dirname`` 用于指定加载变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用filename来指定它

参数:
 - **executor**  (Executor) – 加载变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要加载变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **vars**  (list[Variable]|None) –  要加载的变量的列表。 优先级高于main_program。默认值: None
 - **predicate**  (function|None) – 如果不等于None，当指定main_program， 那么只有 predicate(variable)==True 时，main_program中的变量会被加载。
 - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None

抛出异常：
  - **TypeError** - 如果参数 ``main_program`` 为 None 或为一个非 ``Program`` 的实例
   
返回: None
  
**代码示例**

..  code-block:: python
    
    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"

    # 第一种使用方式 使用 main_program 指定变量
    def name_has_fc(var):
        res = "fc" in var.name
        return res

    prog = fluid.default_main_program()
    fluid.io.load_vars(executor=exe, dirname=path, main_program=prog,
                       vars=None)


    # The second usage: using `vars` to specify variables
    var_list = [var_a, var_b, var_c]
    fluid.io.load_vars(executor=exe, dirname=path, vars=var_list,
                       filename="vars_file")
    # var_a, var_b and var_c will be loaded. And they are supposed to haven
    # been saved in the same file named 'var_file' in the path "./my_paddle_model".
 
.. _cn_api_fluid_io_load_params:

load_params
>>>>>>>>>>>>

.. py:class:: paddle.fluid.io.load_params(executor, dirname, main_program=None, filename=None)

该函数过滤掉 给定 ``main_program`` 中所有参数，然后将它们加载保存在到目录 ``dirname`` 中或文件中的参数。

``dirname`` 用于指定保存变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用filename来指定它

注意:有些变量不是参数，但它们对于训练是必要的。因此，您不能仅通过 ``save_params()`` 和 ``load_params()`` 保存并之后继续训练。可以使用 ``save_persistables()`` 和 ``load_persistables()`` 代替这两个函数

参数:
 - **executor**  (Executor) – 加载变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要加载变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None

返回: None
  
**代码示例**

..  code-block:: python
    
    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.load_params(executor=exe, dirname=param_path,
                        main_program=None)
                        
.. _cn_api_fluid_io_load_persistables:

load_persistables
>>>>>>>>>>>>

.. py:class:: paddle.fluid.io.load_persistables(executor, dirname, main_program=None, filename=None)

该函数过滤掉 给定 ``main_program`` 中所有参数，然后将它们加载保存在到目录 ``dirname`` 中或文件中的参数。

``dirname`` 用于指定保存变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用filename来指定它

参数:
 - **executor**  (Executor) – 加载变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要加载变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
-  **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None

返回: None
  
**代码示例**

..  code-block:: python

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.load_persistables(executor=exe, dirname=param_path,
                               main_program=None)
 
.. _cn_api_fluid_io_save_inference_model:

save_inference_model
>>>>>>>>>>>>

.. py:class:: paddle.fluid.io.save_inference_model(dirname, feeded_var_names, target_vars, executor, main_program=None, model_filename=None, params_filename=None, export_for_deployment=True)

修改指定的 ``main_program`` ，构建一个专门用于推理的 ``Program``，然后  ``executor`` 把它和所有相关参数保存到 ``dirname`` 中

``dirname`` 用于指定保存变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用filename来指定它

参数:
  - **dirname** (str) – 保存推理model的路径
  - **feeded_var_names** (list[str]) – 推理（inference）需要 feed 的数据
  - **target_vars** (list[Variable]) – 保存推理（inference）结果的 Variables
  - **executor** (Executor) –  executor 保存  inference model
  - **main_program** (Program|None) – 使用 ``main_program`` ，构建一个专门用于推理的 ``Program``（inference model）. 如果为None, 使用   ``default main program``   默认: None.
  - **model_filename** (str|None) – 保存 推理P rogram 的文件名称。如果设置为None，将使用默认的文件名为： filename_model__
  - **params_filename** (str|None) – 保存所有相关参数的文件名称。如果设置为None，则参数将保存在单独的文件中。
  - **export_for_deployment** (bool) – 如果为真，Program将被修改为只支持直接推理部署的Program。否则，将存储更多的信息，方便优化和再训练。目前只支持True。

返回: None

抛出异常：
 - **ValueError** – 如果 ``feed_var_names`` 不是字符串列表
 - **ValueError** – 如果 ``target_vars`` 不是 ``Variable`` 列表

**代码示例**

..  code-block:: python

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./infer_model"
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],
                 target_vars=[predict_var], executor=exe)

    # 在这个示例中，函数将修改默认的主程序让它适合于推断‘predict_var’。修改的
    # 推理Program 将被保存在 ./infer_model/__model__”中。
    # 和参数将保存在文件夹下的单独文件中 ./infer_mode


.. _cn_api_fluid_io_load_inference_model:

save_inference_model
>>>>>>>>>>>>

.. py:class:: paddle.fluid.io.load_inference_model(dirname, executor, model_filename=None, params_filename=None, pserver_endpoints=None)

从指定目录中加载 推理model（inference model ）

参数:
  - **dirname** (str) – model的路径
  - **executor** (Executor) – 运行 inference model的 ``executor``
  - **model_filename** (str|None) –  推理 Program 的文件名称。如果设置为None，将使用默认的文件名为： filename_model__
  - **params_filename** (str|None) –  加载所有相关参数的文件名称。如果设置为None，则参数将保存在单独的文件中。
  - pserver_endpoints (list|None) – 只有在分布式推理时需要用到。 当在训练时使用分布式 look up table , 需要这个参数. 该参数是 pserver endpoints 的列表 

返回: 这个函数的返回有三个元素的元组(Program，feed_target_names, fetch_targets)。Program 是一个 ``Program`` ，它是推理 ``Program``。  ``feed_target_names`` 是一个str列表，它包含需要在推理 ``Program`` 中提供数据的变量的名称。` `fetch_targets`` 是一个 ``Variable`` 列表，从中我们可以得到推断结果。

返回类型：元组(tuple)

抛出异常：
 - **ValueError** – 如果 ``dirname`` 非法 

..  code-block:: python

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./infer_model"
    endpoints = ["127.0.0.1:2023","127.0.0.1:2024"]
    [inference_program, feed_target_names, fetch_targets] =
        fluid.io.load_inference_model(dirname=path, executor=exe)
    results = exe.run(inference_program,
                  feed={feed_target_names[0]: tensor_img},
                  fetch_list=fetch_targets)
    # 在这个示例中，inference program 保存在 ./infer_model/__model__”中
    # 参数保存在./infer_mode 单独的若干文件中
    # 加载 inference program 后， executor 使用 fetch_targets 和 feed_target_names 执行Program， 得到推理结果
