.. _cn_api_fluid_io_load_vars:

load_vars
-------------------------------

.. py:function:: paddle.fluid.io.load_vars(executor, dirname, main_program=None, vars=None, predicate=None, filename=None)

该接口从文件中加载 ``Program`` 的变量。

通过 ``vars`` 指定需要加载的变量，或者通过 ``predicate`` 筛选需要加载的变量， ``vars`` 和 ``predicate`` 不能同时为None。

参数:
 - **executor**  (Executor) – 运行的执行器，执行器的介绍请参考 :ref:`api_guide_model_save_reader` 。
 - **dirname**  (str) – 加载变量所在的目录路径。
 - **main_program**  (Program，可选) – 需要加载变量的 ``Program`` ， ``Program`` 的介绍请参考 :ref:`api_guide_Program` 。如果 ``main_program`` 为None，则使用默认的主程序。默认值为None。
 - **vars**  (list[Variable]，可选) –  通过该列表指定需要加载的变量。默认值为None。
 - **predicate**  (function，可选) – 通过该函数筛选 :math:`predicate(variable)== True` 的变量进行加载。如果通过 ``vars`` 指定了需要加载的变量，则该参数无效。默认值为None。
 - **filename**  (str，可选) – 加载所有变量的文件。如果所有待加载变量是保存在一个文件中，则设置 ``filename`` 为该文件名；如果所有待加载变量是按照变量名称单独保存成文件，则设置 ``filename`` 为None。默认值为None。

返回： 无

抛出异常：
  - ``TypeError`` - 如果main_program不是Program的实例，也不是None。
 
  
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

    # 示例一：用vars来指定加载变量。
    path = "./my_paddle_vars"
    var_list = [w, b]
    fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                       filename="vars_file")
    fluid.io.load_vars(executor=exe, dirname=path, vars=var_list,
                       filename="vars_file")
    # 加载w和b。它们被保存在'var_file'的文件中，所在路径为 "./my_paddle_model" 。
    
    # 示例二：通过predicate来筛选加载变量。
    def name_has_fc(var):
        res = "fc" in var.name
        return res
    
    param_path = "./my_paddle_model"
    fluid.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate=name_has_fc)
    fluid.io.load_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate=name_has_fc)
    #加载 `main_program` 中变量名包含 ‘fc’ 的所有变量
    #此前所有变量应该保存在不同文件中

 


