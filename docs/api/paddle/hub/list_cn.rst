.. _cn_api_paddle_hub_list:

list
-------------------------------

.. py:function:: paddle.hub.list(repo_dir, source='github', force_reload=False)


用于查看指定repo提供的功能或者模型列表


参数
:::::::::

    - **repo_dir** （str）: repo地址，支持git地址形式和local地址。git地址由repo拥有者/repo名字:repo分支组成，实例：PaddlePaddle/PaddleClas:develop；local地址为repo的本地路径
    - **source** （str | 可选）: 指定repo托管的位置，支持github和local，默认值：github
    - **force_reload** （bool | 可选） : 指定是否强制拉取，默认值: False

返回
:::::::::

    ``list`` ，repo提供的模型/功能列表


代码示例
:::::::::

.. code-block:: python

    import paddle
    models = paddle.hub.list('PaddlePaddle/PaddleClas:develop', source='github', force_reload=True)    
    print(models)
