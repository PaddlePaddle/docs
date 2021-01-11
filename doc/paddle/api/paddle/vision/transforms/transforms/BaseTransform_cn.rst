.. _cn_api_vision_transforms_BaseTransform:

BaseTransform
-------------------------------

.. py:class:: paddle.vision.transforms.BaseTransform(keys=None)

视觉中图像变化的基类。

调用逻辑: 

.. code-block:: text

    if keys is None:
        _get_params -> _apply_image()
    else:
        _get_params -> _apply_*() for * in keys 

如果你想要定义自己的图像变化方法, 需要重写子类中的 ``_apply_*`` 方法。

参数
:::::::::

    - keys (list[str]|tuple[str], optional) - 输入的类型. 你的输入可以是单一的图像，也可以是包含不同数据结构的元组, ``keys`` 可以用来指定输入类型. 举个例子, 如果你的输入就是一个单一的图像，那么 ``keys`` 可以为 ``None`` 或者 ("image")。如果你的输入是两个图像：``(image, image)`` ，那么 `keys` 应该设置为 ``("image", "image")`` 。如果你的输入是 ``(image, boxes)``, 那么 ``keys`` 应该为 ``("image", "boxes")`` 。

            目前支持的数据类型如下所示:

            - "image": 输入的图像, 它的维度为 ``(H, W, C)`` 。 
            - "coords": 输入的左边, 它的维度为 ``(N, 2)`` 。 
            - "boxes": 输入的矩形框, 他的维度为 (N, 4), 形式为 "xyxy", 第一个 "xy" 表示矩形框左上方的坐标, 第二个 "xy" 表示矩形框右下方的坐标.
            - "mask": 分割的掩码，它的维度为 ``(H, W, 1)`` 。
            
            你也可以通过自定义 _apply_*的方法来处理特殊的数据结构。

返回
:::::::::

    ``PIL.Image 或 numpy ndarray``，变换后的图像。

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image
    import paddle.vision.transforms.functional as F
    from paddle.vision.transforms import BaseTransform

    def _get_image_size(img):
        if F._is_pil_image(img):
            return img.size
        elif F._is_numpy_image(img):
            return img.shape[:2][::-1]
        else:
            raise TypeError("Unexpected type {}".format(type(img)))

    class CustomRandomFlip(BaseTransform):
        def __init__(self, prob=0.5, keys=None):
            super(CustomRandomFlip, self).__init__(keys)
            self.prob = prob

        def _get_params(self, inputs):
            image = inputs[self.keys.index('image')]
            params = {}
            params['flip'] = np.random.random() < self.prob
            params['size'] = _get_image_size(image)
            return params

        def _apply_image(self, image):
            if self.params['flip']:
                return F.hflip(image)
            return image

        # if you only want to transform image, do not need to rewrite this function
        def _apply_coords(self, coords):
            if self.params['flip']:
                w = self.params['size'][0]
                coords[:, 0] = w - coords[:, 0]
            return coords

        # if you only want to transform image, do not need to rewrite this function
        def _apply_boxes(self, boxes):
            idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
            coords = np.asarray(boxes).reshape(-1, 4)[:, idxs].reshape(-1, 2)
            coords = self._apply_coords(coords).reshape((-1, 4, 2))
            minxy = coords.min(axis=1)
            maxxy = coords.max(axis=1)
            trans_boxes = np.concatenate((minxy, maxxy), axis=1)
            return trans_boxes
            
        # if you only want to transform image, do not need to rewrite this function
        def _apply_mask(self, mask):
            if self.params['flip']:
                return F.hflip(mask)
            return mask

    # create fake inputs
    fake_img = Image.fromarray((np.random.rand(400, 500, 3) * 255.).astype('uint8'))
    fake_boxes = np.array([[2, 3, 200, 300], [50, 60, 80, 100]])
    fake_mask = fake_img.convert('L')

    # only transform for image:
    flip_transform = CustomRandomFlip(1.0)
    converted_img = flip_transform(fake_img)

    # transform for image, boxes and mask
    flip_transform = CustomRandomFlip(1.0, keys=('image', 'boxes', 'mask'))
    (converted_img, converted_boxes, converted_mask) = flip_transform((fake_img, fake_boxes, fake_mask))
    print('converted boxes', converted_boxes)
    
