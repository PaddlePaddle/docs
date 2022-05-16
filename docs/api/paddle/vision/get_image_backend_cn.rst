.. _cn_api_vision_image_get_image_backend:

get_image_backend
-------------------------------

.. py:function:: paddle.vision.image.get_image_backend()

获取用于加载图像的模块名称。


返回
:::::::::

    ``str``，加载图像的模块名称， ``pil`` 或 ``cv2``。

代码示例
:::::::::

.. code-block:: python

    from paddle.vision import get_image_backend

    backend = get_image_backend()
    print(backend)
        