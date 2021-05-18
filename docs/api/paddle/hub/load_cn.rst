.. _cn_api_paddle_hub_load:

load
-------------------------------

.. py:function:: paddle.hub.load(repo_dir, model, source='github', force_reload=False, **kwargs)

用于加载repo提供的功能/模型列表


参数
:::::::::

    - **repo_dir** （str）: repo地址，支持git地址形式和local地址。git地址由repo拥有者/repo名字:repo分支组成，实例：PaddlePaddle/PaddleClas:develop；local地址为repo的本地路径
    - **model** （str）: 模型的名字
    - **source** （str | 可选）: 指定repo托管的位置，支持github和local，默认值：github
    - **force_reload** （bool | 可选） : 指定是否强制拉取，默认值: False
    - **kwargs** （any | 可选） : 模型参数

返回
:::::::::

    ``paddle.nn.Layer`` ，repo提供的指定模型实例


代码示例
:::::::::

.. code-block:: python

    import paddle
    model = paddle.hub.help('PaddlePaddle/PaddleClas:develop', 'alexnet', source='github', force_reload=True)    
    print(model)
