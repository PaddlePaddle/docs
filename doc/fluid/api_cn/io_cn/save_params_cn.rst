.. _cn_api_fluid_io_save_params:

save_params
-------------------------------

.. py:function:: paddle.fluid.io.save_params(executor, dirname, main_program=None, filename=None)

该函数从 ``main_program`` 中取出所有参数，然后将它们保存到 ``dirname`` 目录下或名为 ``filename`` 的文件中。

``dirname`` 用于指定保存变量的目标目录。如果想将变量保存到多个独立文件中，设置 ``filename`` 为 None; 如果想将所有变量保存在单个文件中，请使用 ``filename`` 来指定该文件的命名。

注意:有些变量不是参数，但它们对于训练是必要的。因此，调用 ``save_params()`` 和 ``load_params()`` 来保存和加载参数是不够的，可以使用 ``save_persistables()`` 和 ``load_persistables()`` 代替这两个函数。如果您想要储存您的模型用于预测，请使用save_inference_model API。更多细节请参考 :ref:`api_guide_model_save_reader`。


参数:
 - **executor**  (Executor) – 保存变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要保存变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **vars**  (list[Variable]|None) –  要保存的所有变量的列表。 优先级高于main_program。默认值: None
 - **filename**  (str|None) – 保存变量的文件。如果想分不同独立文件来保存变量，设置 filename=None. 默认值: None
 
返回: None
  
**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.save_params(executor=exe, dirname=param_path,
                         main_program=None)
                         






