.. _cn_api_fluid_io_load_vars:

load_vars
-------------------------------

.. py:function:: paddle.fluid.io.load_vars(executor, dirname, main_program=None, vars=None, predicate=None, filename=None)

``executor`` 从指定目录加载变量。

有两种方法来加载变量:方法一，``vars`` 为变量的列表。方法二，将已存在的 ``Program`` 赋值给 ``main_program`` ，然后将加载 ``Program`` 中的所有变量。第一种方法优先级更高。如果指定了 vars，那么忽略 ``main_program`` 和 ``predicate`` 。

``dirname`` 用于指定加载变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用 ``filename`` 来指定它。

参数:
 - **executor**  (Executor) – 加载变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要加载变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **vars**  (list[Variable]|None) –  要加载的变量的列表。 优先级高于main_program。默认值: None
 - **predicate**  (function|None) – 如果不等于None，当指定main_program， 那么只有 predicate(variable)==True 时，main_program中的变量会被加载。
 - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None

抛出异常：
  - ``TypeError`` - 如果参数 ``main_program`` 为 None 或为一个非 ``Program`` 的实例
   
返回: None
  
**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    main_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(main_prog, startup_prog):
        data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
        w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
        b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
        hidden_w = fluid.layers.matmul(x=data, y=w)
        hidden_b = fluid.layers.elementwise_add(hidden_w, b)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    param_path = "./my_paddle_model"

    # 第一种使用方式 使用 main_program 指定变量
    def name_has_fc(var):
        res = "fc" in var.name
        return res
    fluid.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate=name_has_fc)
    fluid.io.load_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate=name_has_fc)
    #加载所有`main_program`中变量名包含 ‘fc’ 的变量
    #并且此前所有变量应该保存在不同文件中

    #用法2：使用 `vars` 来使变量具体化
    path = "./my_paddle_vars"
    var_list = [w, b]
    fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                       filename="vars_file")
    fluid.io.load_vars(executor=exe, dirname=path, vars=var_list,
                       filename="vars_file")
    # 加载w和b，它们此前应被保存在同一名为'var_file'的文件中
    # 该文件所在路径为 "./my_paddle_model"
 


