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




