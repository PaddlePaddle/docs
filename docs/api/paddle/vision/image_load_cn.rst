.. _cn_api_vision_image_image_load:

image_load
-------------------------------

.. py:function:: paddle.vision.image.image_load(path, backend=None)

读取一个图像。

参数
:::::::::

    - path (str) - 图像路径。
    - backend (str，可选): 加载图像的后端。 参数必须是 ``cv2``， ``pil``， ``None`` 之一。如果后端为 ``None`` ，则使用全局的 ``_imread_backend`` 参数，默认值为 ``pil`` 。 这个参数可以使用 ``paddle.vision.set_image_backend`` 指定。 默认值：None 。

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray``，加载后的图像。

代码示例
:::::::::

.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision import image_load, set_image_backend

    fake_img = Image.fromarray((np.random.random((32, 32, 3)) * 255).astype('uint8'))

    path = 'temp.png'
    fake_img.save(path)

    set_image_backend('pil')
    
    pil_img = image_load(path).convert('RGB')

    # should be PIL.Image.Image
    print(type(pil_img))

    # use opencv as backend
    # set_image_backend('cv2')

    # np_img = image_load(path)
    # # should get numpy.ndarray
    # print(type(np_img))
        
