.. _cn_api_fluid_io_save_vars:

save_vars
>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.io.save_vars(executor, dirname, main_program=None, vars=None, predicate=None, filename=None)

通过执行此函数将变量保存到指定目录下。

有两种方法可以指定要保存的变量：第一种方法，列出列表中的变量并将其分配给变量。第二种方法是，将 ``main_program`` 分配给现有程序，然后保存程序中的所有变量。第一种方式具有更高的优先级。换句话说，如果分配了变量，则将忽略 ``main_program`` 和 ``predicate`` 。

 ``dirname`` 用于指定保存变量的文件夹。如果您希望将变量保存在文件夹目录的单独文件中，请设置文件名为无；如果您希望将所有变量保存在单个文件中，请使用filename指定它。

参数：
      - **executor** （Executor）:为保存变量而运行的执行程序。
      - **dirname** （str）:目录路径。
      - **main_program** （Program | None）：保存变量的程序。如果为None，将自动使用默认主程序。默认值：None。
      - **vars**（list [Variable] | None）：包含要保存的所有变量的列表。它的优先级高于 ``main_program`` 。默认值：None。
      - **predicate**（function | None）：如果它不是None，则只保存 ``main_program`` 中使 :math:`predicate(variable)== True` 的变量。它仅在我们使用 ``main_program`` 指定变量时才起作用（换句话说，vars为None）。默认值：None。
      - **filename** （str | None）：保存所有变量的文件。如果您希望单独保存变量，请将其设置为“None”。默认值：None。

返回：     None

抛出：    TypeError - 如果main_program不是Program的实例，也不是None。

**代码示例**

..  code-block:: python

      exe = fluid.Executor(fluid.CPUPlace())
      param_path = "./my_paddle_model"

      # 第一种用法:用main_program来指定变量。
      def name_has_fc(var):
          res = "fc" in var.name
          return res

      prog = fluid.default_main_program()
      fluid.io.save_vars(executor=exe, dirname=path, main_program=prog,
                         vars=None)
      # 将main_program中名为“fc”的的所有变量保存。
      # 变量将分开保存。


      # 第二种用法: 用vars来指定变量。
      var_list = [var_a, var_b, var_c]
      fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                         filename="vars_file")
      # var_a，var_b和var_c将被保存。 他们将以相同的文件名“var_file”保存在路径“./my_paddle_model”下。
    
