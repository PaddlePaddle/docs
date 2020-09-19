.. _cn_api_fluid_layers_while_loop:

while_loop
____________________________________



.. py:function:: paddle.fluid.layers.while_loop(cond, body, loop_vars, is_test=False, name=None)




该API用于实现类似while的循环控制功能，只要循环条件 ``cond`` 的返回值为True，``while_loop`` 则会循环执行循环体 ``body`` ，直到 ``cond`` 的返回值为False。

**注意：**
    ``body`` 中定义的局部变量无法使用 ``Executor`` 的 ``fetch_list`` 来获取的，变量需在 ``body`` 外定义并将其置于 ``loop_vars`` 中进行循环更新后才可通过 ``fetch_list`` 获取。

参数：
    - **cond** (callable) - 返回boolean类型张量的可调用函数，用以判断循环是否继续执行。 ``cond`` 的参数和 ``loop_vars`` 相对应。
    - **body** (callable) - 循环执行的结构体。其返回一个包含tensor或LoDTensorArray的列表或元组，且这些tensor或LoDTensorArray的长度，结构，类型和 ``loop_vars`` 中的相同。 且``body`` 的参数与 ``loop_vars`` 相对应。
    - **loop_vars** (list|tuple) - 包含tensor或LoDTensorArray的列表或是元组，将其传入至 ``cond`` 和 ``body`` 中，得到循环条件和输出值。
    - **is_test** (bool，可选) - 用于表明是否在测试阶段执行，默认值为False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`。一般无需设置，默认值为None。

返回：循环迭代之后 ``body`` 的返回值，和 ``loop_vars`` 具有相同的结构。

返回类型：list|tuple

异常抛出：
    - ``TypeError``：若 ``cond`` 不是可调用的。
    - ``TypeError``：若 ``body`` 不是可调用的。
    - ``TypeError``：若 ``loop_vars`` 不是列表或是元组。
    - ``TypeError``：若 ``cond`` 返回值不是变量。
    - ``TypeError``：若 ``cond`` 返回值不是boolean类型的variable。
    - ``TypeError``：若 ``cond`` 返回值的shape不等于1。
    - ``ValueError``：若 ``loop_vars`` 为空。
    - ``ValueError``：若 ``cond`` 返回值的长度和类型和 ``loop_vars`` 不同。

**示例代码**

.. code-block:: python
    
    # 该示例代码展示整数循环+1，循环10次，输出计数结果
    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    
    def cond(i, ten):   # 参数和loop_vars相对应
        return i < ten
    
    def body(i, ten):   # 参数和loop_vars相对应
        i = i + 1
        return [i, ten]
            
    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()
    with fluid.program_guard(main_program, startup_program):
        i = layers.fill_constant(shape=[1], dtype='int64', value=0)     # 循环计数器
        ten = layers.fill_constant(shape=[1], dtype='int64', value=10)  # 循环次数
        i, ten = layers.while_loop(cond, body, [i, ten])
                
        exe = fluid.Executor(fluid.CPUPlace())
        res = exe.run(main_program, feed={}, fetch_list=[i])
        print(res) #[array([10])]
