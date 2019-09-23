.. _cn_api_fluid_io_save_vars:

save_vars
-------------------------------

.. py:function:: paddle.fluid.io.save_vars(executor, dirname, main_program=None, vars=None, predicate=None, filename=None)

该接口将程序中的变量保存到指定目录。

通过 ``vars`` 指定需要保存的变量，或者通过 ``predicate`` 筛选需要保存的变量， ``vars`` 和 ``predicate`` 不能同时为None。

参数：
      - **executor** （Executor）- 运行的执行器。
      - **dirname** （str）- 保存变量的目录路径。
      - **main_program** （Program，可选）- 需要保存变量的程序。如果为None，则使用默认的主程序。默认值为None。
      - **vars** （list [Variable]，可选）- 通过该列表指定需要保存的变量。默认值为None。
      - **predicate** （function，可选）- 通过该函数筛选 :math:`predicate(variable)== True` 的变量进行保存。如果通过 ``vars`` 指定了需要保存的变量，则该参数无效。默认值为None。
      - **filename** （str，可选）- 保存所有变量的文件。如果设置为None，所有变量会按照变量名称单独保存成文件；如果设置为非None，所有变量会保存成一个文件名为该设置值的文件。默认值为None。

返回：无    

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
     
      # 用vars来指定变量。
      var_list = [w, b]
      path = "./my_paddle_vars"
      fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                         filename="vars_file")
      # w, b 将被保存，使用同一文件名“var_file”，保存在路径“./my_paddle_vars”下。

      # 通过predicate筛选变量。
      def name_has_fc(var):
          res = "fc" in var.name
          return res

      param_path = "./my_paddle_model"
      fluid.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate = name_has_fc)
      # 将main_program中名中包含“fc”的的所有变量保存。
      # 变量将分开保存。





