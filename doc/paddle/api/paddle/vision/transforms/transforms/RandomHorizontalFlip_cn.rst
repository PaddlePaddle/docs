.. _cn_api_vision_transforms_RandomHorizontalFlip:

RandomHorizontalFlip
-------------------------------

.. py:class:: paddle.vision.transforms.RandomHorizontalFlip(prob=0.5)

基于概率来执行图片的水平翻转。

参数
:::::::::

    - prob (float) - 图片执行水平翻转的概率，默认值为0.5。

返回
:::::::::

    ``numpy ndarray``，概率执行水平翻转后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from paddle.vision.transforms import RandomHorizontalFlip

    transform = RandomHorizontalFlip(1)
    np.random.seed(5)
    fake_img = np.random.rand(3, 3, 3).astype('float32')
    print('翻转前的图片')
    print(fake_img)
    fake_img = transform(fake_img)

    print('翻转后的图片')
    print(fake_img)
    """
    翻转前的图片
    [[[0.22199318 0.8707323  0.20671916]
    [0.91861093 0.4884112  0.61174387]
    [0.7659079  0.518418   0.2968005 ]]

    [[0.18772122 0.08074127 0.7384403 ]
    [0.4413092  0.15830986 0.87993705]
    [0.27408648 0.41423503 0.29607993]]

    [[0.62878793 0.5798378  0.5999292 ]
    [0.26581913 0.28468588 0.2535882 ]
    [0.32756394 0.1441643  0.16561286]]]
    翻转后的图片
    [[[0.7659079  0.518418   0.2968005 ]
    [0.91861093 0.4884112  0.61174387]
    [0.22199318 0.8707323  0.20671916]]

    [[0.27408648 0.41423503 0.29607993]
    [0.4413092  0.15830986 0.87993705]
    [0.18772122 0.08074127 0.7384403 ]]

    [[0.32756394 0.1441643  0.16561286]
    [0.26581913 0.28468588 0.2535882 ]
    [0.62878793 0.5798378  0.5999292 ]]]
    """