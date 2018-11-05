.. _cn_api_fluid_default_startup_program:




default_startup_program
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

paddle.fluid.default_startup_program()
""""""""""""""""""""""""""""""""""""""""""


该函数可以获取默认/全局 startup program (启动程序)。

``fluid.layers`` 中的layer函数会新建参数、readers(读取器)、NCCL句柄作为全局变量。 

startup_program会使用内在的operators（算子）去初始化他们，并由layer函数将这些operators追加到startup program中。

该函数将返回默认的或当前的startup_program。用户可以使用 ``fluid.program_guard`` 去切换program。

返回:	startup program

返回类型:	Program





.. _cn_api_fluid_default_main_program:

default_main_program
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

paddle.fluid.default_main_program()
""""""""""""""""""""""""""""""""""""""""""




此函数用于获取默认或全局main program(主程序)。该主程序用于训练和测试模型。

``fluid.layers`` 中的所有layer函数可以向 ``default_main_program`` 中添加operators（算子）和variables（变量）。

``default_main_program`` 是fluid的许多编程接口（API）的Program参数的缺省值。例如,当用户program没有传入的时候，
``Executor.run()`` 会默认执行 ``default_main_program`` 。


返回：	main program

返回类型:	Program





.. _cn_api_fluid_program_guard:



program_guard
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

paddle.fluid.program_guard(*args, **kwds)
""""""""""""""""""""""""""""""""""""""""""


该函数应配合使用python的“with”语句来改变全局主程序(main program)和启动程序(startup program)。

“with”语句块中的layer函数将在新的main program（主程序）后添加operators（算子）和variables（变量）。

**代码示例**

..  code-block:: python

	import paddle.fluid as fluid
	main_program = fluid.Program()
	startup_program = fluid.Program()
	with fluid.program_guard(main_program, startup_program):
		data = fluid.layers.data(...)
 		hidden = fluid.layers.fc(...)

需要注意的是，如果用户不需要构建自己的启动程序或者主程序，一个临时的program将会发挥作用。



**代码示例**

..  code-block:: python

	import paddle.fluid as fluid
	main_program = fluid.Program()
	# does not care about startup program. Just pass a temporary value.
	with fluid.program_guard(main_program, fluid.Program()):
		data = ...


参数：  
		- **main_program** (Program) – “with”语句中将使用的新的main program。
		- **startup_program** (Program) – “with”语句中将使用的新的startup program。若传入 ``None`` 则不改变当前的启动程序。



