.. _cn_api_fluid_io_save_params:

save_params
-------------------------------


.. py:function:: paddle.fluid.io.save_params(executor, dirname, main_program=None, filename=None)




该OP从 ``main_program`` 中取出所有参数，然后将它们保存到 ``dirname`` 目录下或名为 ``filename`` 的文件中。

``dirname`` 用于指定保存参数的目标路径。若想将参数保存到多个独立文件中，设置 ``filename=None`` ； 若想将所有参数保存在单个文件中，请设置 ``filename`` 来指定该文件的名称。

注意：
   - 有些变量不是参数，如学习率，全局训练步数（global step）等，但它们对于训练却是必要的。因此，调用 :ref:`cn_api_fluid_io_save_params` 和 :ref:`cn_api_fluid_io_load_params` 来保存和加载参数对于断点训练是不够的，这种情况下可以使用 :ref:`cn_api_fluid_io_save_persistables` 和 :ref:`cn_api_fluid_io_load_persistables` 来保存和加载训练过程中的检查点（checkpoint）。
   - 如果您想要储存您的模型用于预测，请使用 :ref:`cn_api_fluid_io_save_inference_model` 。更多细节请参考 :ref:`api_guide_model_save_reader` 

参数:
 - **executor**  (Executor) – 用于保存参数的 ``executor`` ，详见 :ref:`api_guide_executor` 。
 - **dirname**  (str) – 指定保存参数的文件目录。
 - **main_program**  (Program，可选) – 需要保存参数的Program（ ``Program`` 含义详见 :ref:`api_guide_Program` ）。如果为None，则使用default_main_Program 。默认值为None。
 - **filename**  (str，可选) – 保存参数的文件名称。若需要将参数保存到多个独立的文件中，请设置 ``filename=None`` 。默认值为None。
 
返回: 无
  
**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    
    params_path = "./my_paddle_model"
    image = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
    predict = fluid.layers.fc(input=image, size=10, act='softmax')

    loss = fluid.layers.cross_entropy(input=predict, label=label)
    avg_loss = fluid.layers.mean(loss)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    fluid.io.save_params(executor=exe, dirname=params_path)
    # 网络中fc层的参数weight和bias将会分别存储在"./my_paddle_model"路径下。                    

