.. _cn_api_fluid_dygraph_load_dygraph:

load
----


.. py:function:: paddle.fluid.dygraph.load_dygraph(model_path, **configs)


该接口用于从磁盘中加载Layer和Optimizer的 ``state_dict``，该接口会同时加载 ``model_path + ".pdparams"`` 和 ``model_path + ".pdopt"`` 中的内容。

.. note::
    如果从 ``paddle.static.save_inference_model`` 的存储结果中载入 ``state_dict``，动态图模式下参数的结构性变量名将无法被恢复。并且在将载入的 ``state_dict`` 配置到当前Layer中时，需要配置 ``Layer.set_state_dict`` 的参数 ``use_structured_name=False`` 。

参数
:::::::::
    - model_path (str) – 保存state_dict的文件前缀。该路径不应该包括后缀 ``.pdparams`` 或 ``.pdopt``。
    - **config (dict，可选) - 其他用于兼容的载入配置选项。这些选项将来可能被移除，如果不是必须使用，不推荐使用这些配置选项。默认为 ``None``。目前支持以下配置选项：(1) model_filename (str) - paddle 1.x版本 ``save_inference_model`` 接口存储格式的预测模型文件名，原默认文件名为 ``__model__`` ； (2) params_filename (str) - paddle 1.x版本 ``save_inference_model`` 接口存储格式的参数文件名，没有默认文件名，默认将各个参数分散存储为单独的文件。

返回
:::::::::
tuple(dict, dict)，两个 ``dict``，即从文件中恢复的模型参数 ``dict`` 和优化器参数 ``dict``，如果只找到其中一个的存储文件，另一个返回None

- param_dict：从文件中恢复的模型参数 ``dict``
- opt_dict：从文件中恢复的优化器参数 ``dict``
  
代码示例
:::::::::

COPY-FROM: paddle.fluid.dygraph.load_dygraph