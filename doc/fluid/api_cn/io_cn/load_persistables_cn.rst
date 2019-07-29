.. _cn_api_fluid_io_load_persistables:

load_persistables
-------------------------------

.. py:function:: paddle.fluid.io.load_persistables(executor, dirname, main_program=None, filename=None)

该函数从给定 ``main_program`` 中取出所有 ``persistable==True`` 的变量（即长期变量），然后将它们从目录 ``dirname`` 中或 ``filename`` 指定的文件中加载出来。

``dirname`` 用于指定存有长期变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用filename来指定它。

参数:
    - **executor**  (Executor) – 加载变量的 executor
    - **dirname**  (str) – 目录路径
    - **main_program**  (Program|None) – 需要加载变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
    - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None

返回: None
  
**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.load_persistables(executor=exe, dirname=param_path,
                               main_program=None)
 






