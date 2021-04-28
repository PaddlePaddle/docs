.. _cn_api_paddle_hub_help:

help
-------------------------------

.. py:function:: paddle.hub.help(repo_dir, model, source='github', force_reload=False)


用于查看repo提供的功能/模型的文档


参数
:::::::::

    - **repo_dir** （str）: repo地址。由repo拥有者/repo名字:repo分支组成，实例：PaddlePaddle/PaddleClas:develop
    - **model** （str）: 模型的名字
    - **source** （str | 可选）: 指定repo托管的位置，默认值：github
    - **force_reload** （bool | 可选） : 指定是否强制拉取，默认值: False

返回
:::::::::

    ``str`` ，repo提供的指定模型的文档


代码示例
:::::::::

.. code-block:: python

    import paddle
    docs = paddle.hub.help('PaddlePaddle/PaddleClas:develop', source='alexnet', force_reload=True)    
    print(docs)
