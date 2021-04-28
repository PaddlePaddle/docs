.. _cn_api_paddle_hub_repos_paddleClas:

paddleClas
-------------------------------

paddleClas支持模型列表
::::::::::::::::::::

.. csv-table::
    :header: "模型名字", "功能"
    :widths: 10, 30

    " :ref:`axlenet <cn_api_paddle_hub_repos_paddleClas>` ", "axlenet模型"
    " :ref:`axlenet <cn_api_paddle_hub_repos_paddleClas>` ", "axlenet模型"



代码示例
:::::::::

.. code-block:: python

    import paddle
    model_names = paddle.hub.list('PaddlePaddle/PaddleClas:develop', source='github', force_reload=True)    
    model_docs = paddle.hub.help('PaddlePaddle/PaddleClas:develop', 'alexnet', source='github', force_reload=True)    
    model = paddle.hub.list('PaddlePaddle/PaddleClas:develop', 'alexnet', source='github', force_reload=True)    
