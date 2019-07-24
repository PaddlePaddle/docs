.. _cn_api_fluid_transpiler_memory_optimize:

memory_optimize
-------------------------------

.. py:function:: paddle.fluid.transpiler.memory_optimize(input_program, skip_opt_set=None, print_log=False, level=0, skip_grads=False)

历史遗留内存优化策略，通过在不同operators之间重用可变内存来减少总内存消耗。
用一个简单的例子来解释该算法：

c = a + b  # 假设此处是最后一次使用a
d = b * c

因为在“c = a + b”之后将不再使用a，并且a和d的大小相同，所有我们可以使用变量a来替换变量d，即实际上我们可以将上面的代码优化为如下所示：

c = a + b
a = b * c

请注意，在这个历史遗留设计中，我们使用变量a直接替换d，这意味着在调用此API之后，某些变量可能会消失，而某些变量可能会保留非预期值，如在上面的例子中，实际上执行代码后a保持d的值。

因此，为了防止重要变量在优化中被重用/删除，我们提供skip_opt_set用于指定变量白名单。
skip_opt_set中的变量不受memory_optimize API的影响。

注意：
此API已弃用，请避免在新代码中使用它。
不支持会创建子块的运算符，如While，IfElse等。

参数:
  - **input_program** (str) – 输入Program。
  - **skip_opt_set** (set) – set中的vars将不被内存优化。
  - **print_log** (bool) – 是否打印debug日志。
  - **level** (int) - 0或1，0代表我们仅在a.size == b.size时用b替换a，1代表我们可以在a.size <= b.size时用b替换a

返回: None

**代码示例**

.. code-block:: python

          import paddle.fluid as fluid
          main_prog = fluid.Program()
          startup_prog = fluid.Program()
           
          place = fluid.CPUPlace()
          exe = fluid.Executor(place)
           
          exe.run(startup_prog)
          fluid.memory_optimize(main_prog)




