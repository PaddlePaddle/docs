.. _cn_api_paddle_hub_repos_paddleClas:

paddleClas
-------------------------------

模型列表
::::::::::::::::::::

.. csv-table::
    :header: "模型名字"
    :widths: 10

    " :ref:`alexnet` "
    " :ref:`vgg13` "


代码示例
:::::::::

.. code-block:: python

    import paddle
    model_names = paddle.hub.list('PaddlePaddle/PaddleClas:develop', source='github', force_reload=True)    
    model_docs = paddle.hub.help('PaddlePaddle/PaddleClas:develop', 'alexnet', source='github', force_reload=True)    
    model = paddle.hub.load('PaddlePaddle/PaddleClas:develop', 'alexnet', source='github', force_reload=True)    
