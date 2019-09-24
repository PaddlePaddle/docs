.. _cn_api_fluid_io_save_persistables:

save_persistables
-------------------------------

.. py:function:: paddle.fluid.io.save_persistables(executor, dirname, main_program=None, filename=None)

该OP从给定 ``main_program`` 中取出所有 ``persistable==True`` 的变量，然后将它们保存到目录 ``dirname`` 中或 ``filename`` 指定的文件中。

**注意：dirname 指定保存长期变量的目录。如果想将长期变量保存到指定目录下的若干文件中，请设置filename为None ; 若需要将所有长期变量保存在一个单独的文件中，请设置filename来指定该文件的名称。**

参数:
 - **executor**  (Executor) – 用于保存长期变量的 ``executor`` ，详见 :ref:`api_guide_executor` 。
 - **dirname**  (str) – 用于储存长期变量的文件目录。
 - **main_program**  (Program|None) – 需要保存长期变量的 Program。如果为None，则使用default_main_Program 。默认值为None。
 - **filename**  (str|None) – 保存长期变量的文件名。若要分开保存变量，设置filename=None。 默认值为None。
 
返回: 无
  
**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    # `prog` 可以是由用户自定义的program
    fluid.io.save_persistables(executor=exe, dirname=param_path,
                               main_program=prog)

