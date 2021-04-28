.. _cn_api_paddle_hub_repos_paddleNLP:

paddleNLP
-------------------------------

模型列表

::::::::::::::::::::

.. csv-table::
    :header: "模型名字", "功能"
    :widths: 10, 30

    " :ref:`bert <about_paddlenlp_bert>` ", "bert模型"
    

代码示例
:::::::::

.. code-block:: python

    import paddle
    model_names = paddle.hub.list('PaddlePaddle/PaddleNLP:develop', source='github', force_reload=True)    
    model_docs = paddle.hub.help('PaddlePaddle/PaddleNLP:develop', 'bert', source='github', force_reload=True)    
    model = paddle.hub.load('PaddlePaddle/PaddleNLP:develop', 'bert', source='github', force_reload=True)    
    