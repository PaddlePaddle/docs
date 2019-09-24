.. _cn_api_fluid_io_save_params:

save_params
-------------------------------

.. py:function:: paddle.fluid.io.save_params(executor, dirname, main_program=None, filename=None)

该OP从 ``main_program`` 中取出所有参数，然后将它们保存到 ``dirname`` 目录下或名为 ``filename`` 的文件中。

``dirname`` 用于指定保存参数的目标目录。若需要将参数保存到多个独立文件中，请设置 ``filename=None`` ；若需要将所有参数保存在一个单独文件中，请使用 ``filename`` 来指定该文件的命名。

注意:有些变量不是参数，但它们对于训练是必要的。因此，调用 ``save_params()`` 和 ``load_params()`` 来保存和加载参数是不够的，可以使用 ``save_persistables()`` 和 ``load_persistables()`` 代替这两个函数。如果您想要储存您的模型用于预测，请使用save_inference_model API。更多细节请参考 :ref:`api_guide_model_save_reader` 。


参数:
 - **executor**  (Executor) – 用于保存参数的 ``executor`` ，详见 :ref:`api_guide_executor` 。
 - **dirname**  (str) – 指定保存参数的文件目录。
 - **main_program**  (Program|None) – 需要保存参数的Program。如果为 None，则使用 default_main_Program 。默认值为None。
 - **filename**  (str|None) – 保存参数的文件的文件名。若想将参数储存在多个独立的文件中，请设置 ``filename=None`` 。 默认值为None。
 
返回: None
  
**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.save_params(executor=exe, dirname=param_path,
                         main_program=None)
                         






