.. _cn_api_fluid_memory_optimize:

memory_optimize
-------------------------------

.. py:function:: paddle.fluid.memory_optimize(input_program, skip_opt_set=None, print_log=False, level=0, skip_grads=False)

历史遗留的内存优化策略，通过在不同operators间重用var内存来减少总内存消耗。
用一个简单的示例来解释该算法：

c = a + b  # 假设这里是最后一次使用a
d = b * c

鉴于在“c = a + b”之后不再使用a，且a和d的大小相同，我们可以用变量a来代替变量d，即实际上，上面的代码可以优化成：

c = a + b
a = b * c
     
请注意，在此历史遗存设计中，我们将直接用变量a代替变量d，这意味着在你调用该API后，某些变量将会消失，还有一些会取非预期值。正如上面的例子中，执行程序后，实际上a取d的值。
    
因此，为避免重要变量在优化过程中被重用或移除，我们支持用skip_opt_set指定一个变量白名单。skip_opt_set中的变量不会受memory_optimize API的影响。
     
     
.. note::
    
     此API已被弃用，请不要在你新写的代码中使用它。它不支持block中嵌套子block，如While、IfElse等。

参数:
  - **input_program** (str) – 输入Program。
  - **skip_opt_set** (set) – set中的vars将不被内存优化。
  - **print_log** (bool) – 是否打印debug日志。
  - **level** (int) - 值为0或1。如果level=0，则仅当a.size == b.size时我们才用b代替a；如果level=1，只要a.size <= b.size时我们就可以用b代替a。

返回: None

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    main_prog = fluid.Program()
    startup_prog = fluid.Program()
     
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
     
    exe.run(startup_prog)
    fluid.memory_optimize(main_prog)




