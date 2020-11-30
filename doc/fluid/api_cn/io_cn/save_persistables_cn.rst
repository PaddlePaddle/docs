.. _cn_api_fluid_io_save_persistables:

save_persistables
-------------------------------


.. py:function:: paddle.fluid.io.save_persistables(executor, dirname, main_program=None, filename=None)

:api_attr: 声明式编程模式（静态图)



该OP从给定 ``main_program`` 中取出所有持久性变量（详见 :ref:`api_guide_model_save_reader` ），然后将它们保存到目录 ``dirname`` 中或 ``filename`` 指定的文件中。

``dirname`` 用于指定保存持久性变量的目录。如果想将持久性变量保存到指定目录的若干文件中，请设置 ``filename=None`` ; 若想将所有持久性变量保存在同一个文件中，请设置 ``filename`` 来指定文件的名称。

参数:
 - **executor**  (Executor) – 用于保存持久性变量的 ``executor`` ，详见 :ref:`api_guide_executor` 。
 - **dirname**  (str) – 用于储存持久性变量的文件目录。
 - **main_program**  (Program，可选) – 需要保存持久性变量的Program（ ``Program`` 含义详见 :ref:`api_guide_Program` ）。如果为None，则使用default_main_Program 。默认值为None。
 - **filename**  (str，可选) – 保存持久性变量的文件名称。若想分开保存变量，设置 ``filename=None`` 。 默认值为None。
 
返回: 无
  
**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid

    dir_path = "./my_paddle_model"
    file_name = "persistables"
    image = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
    predict = fluid.layers.fc(input=image, size=10, act='softmax')

    loss = fluid.layers.cross_entropy(input=predict, label=label)
    avg_loss = fluid.layers.mean(loss)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    fluid.io.save_persistables(executor=exe, dirname=dir_path, filename=file_name)
    # 网络中fc层中的持久性变量weight和bia将会保存在路径“./my_paddle_model”下名为"persistables"的文件中。
    






