.. _cn_api_fluid_dygraph_load_persistables:

load_persistables
-------------------------------

.. py:function:: paddle.fluid.dygraph.load_persistables(dirname='save_dir')

该函数尝试从dirname中加载持久性变量。


参数:
    - **dirname**  (str) – 目录路径。默认为save_dir


返回:   两个字典:从文件中恢复的参数字典;从文件中恢复的优化器字典

返回类型:   dict
  
**代码示例**

.. code-block:: python

    my_layer = layer(fluid.Layer)
    param_path = "./my_paddle_model"
    sgd = SGDOptimizer(learning_rate=1e-3)
    param_dict, optimizer_dict = fluid.dygraph.load_persistables(my_layer.parameters(), param_path)
    param_1 = param_dict['PtbModel_0.w_1']
    sgd.load(optimizer_dict)



