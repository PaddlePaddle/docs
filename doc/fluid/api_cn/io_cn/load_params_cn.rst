.. _cn_api_fluid_io_load_params:

load_params
-------------------------------

.. py:function:: paddle.fluid.io.load_params(executor, dirname, main_program=None, filename=None)

该函数从给定 ``main_program`` 中取出所有参数，然后从目录 ``dirname`` 中或 ``filename`` 指定的文件中加载这些参数。

``dirname`` 用于存有变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用filename来指明这个文件。

注意:有些变量不是参数，但它们对于训练是必要的。因此，调用 ``save_params()`` 和 ``load_params()`` 来保存和加载参数是不够的，可以使用 ``save_persistables()`` 和 ``load_persistables()`` 代替这两个函数。

如果您想下载预训练后的模型结构和参数用于预测，请使用load_inference_model API。更多细节请参考 :ref:`api_guide_model_save_reader`。

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
    fluid.io.load_params(executor=exe, dirname=param_path,
                        main_program=None)







