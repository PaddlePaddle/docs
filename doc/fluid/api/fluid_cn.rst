.. cn_api_fluid_default_startup_program






default_startup_program
>>>>>>>>>>>>

paddle.fluid.default_startup_program()
""""""""""""""""""""""""""""""""""""""""""
.. 英文原文，方便对照：
.. Get default/global startup program.

.. The layer function in fluid.layers will create parameters, readers, NCCL handles as global variables. The startup_program 
.. will initialize them by the operators in startup program. The layer function will append these initialization operators into startup program.

.. This method will return the default or the current startup program. Users can use fluid.program_guard to switch program.
.. 返回:	startup program
.. 返回类型:	Program


获取默认/全局 startup program (启动程序)

``fluid.layers`` 中的layer函数会新建参数、读取器(readers)、NCCL句柄作为全局变量。 

startup_program会使用内在的operators去初始化他们，并由layer函数将这些operators追加到startup promgram中。

该函数将返回默认的或当前的startup_program。用户可以使用 ``fluid.program_guard`` 去切换program。

返回:	startup program

返回类型:	Program





.. cn_api_fluid_default_main_program_cn

default_main_program
>>>>>>>>>>>>

paddle.fluid.default_main_program()
""""""""""""""""""""""""""""""""""""""""""

.. 英语部分
.. Get default/global main program. The main program is used for training or testing.

.. All layer function in fluid.layers will append operators and variables to the default_main_program.

.. The default_main_program is the default program in a lot of APIs. For example, the Executor.run() will execute 
.. the default_main_program when the program is not specified.
.. 返回:	main program
.. 返回类型:	Program


此函数用于获取默认或全局main program(主程序)。该主程序用于训练和测试模型。

``fluid.layers`` 中的所有layer函数可以向 ``default_main_program`` 中添加算子(operators)和变量(variables)。

``default_main_program`` 是fluid的许多编程接口（API）的默认程序(program)。例如,当用户program未设置的时候，
``Executor.run()`` 会默认执行 ``default_main_program`` 。


返回：	main program

返回类型:	Program







.. cn_api_fluid_program_guard

program_guard
>>>>>>>>>>>>

paddle.fluid.program_guard(*args, **kwds)
""""""""""""""""""""""""""""""""""""""""""
.. Change the global main program and startup program with with statement. 
.. Layer functions in the Python with block will append operators and variables to the new main programs.

使用python的“with”语句改变全局主程序(main program)和启动程序(startup program)。

“with”语句块中的layer函数将在新的主程序（main program）后添加算子（operators）和变量（variables）。

**代码示例**

..  code-block:: python

 >>> import paddle.fluid as fluid
 >>> main_program = fluid.Program()
 >>> startup_program = fluid.Program()
 >>> with fluid.program_guard(main_program, startup_program):
 >>>    data = fluid.layers.data(...)
 >>>    hidden = fluid.layers.fc(...)

需要注意的是，如果用户不需要构建自己的启动程序或者主程序，一个临时的program将会发挥作用。

.. The temporary Program can be used if the user does not need to construct either of startup program or main program.

**代码示例**

..  code-block:: python

>>> import paddle.fluid as fluid
>>> main_program = fluid.Program()
>>> # does not care about startup program. Just pass a temporary value.
>>> with fluid.program_guard(main_program, fluid.Program()):
>>>     data = ...


参数：  
		- **main_program** (Program) – “with”语句中将使用的新的主程序(main program)。
		- **startup_program** (Program) – “with”语句中将使用的新的启动程序(startup program)。若传入 ``None`` 则不改变当前的启动程序。



