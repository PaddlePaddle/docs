.. _cn_api_paddle_vision_datasets_DatasetFolder:

DatasetFolder
-------------------------------

.. py:class:: paddle.vision.datasets.DatasetFolder(root, loader=None, extensions=None, transform=None, is_valid_file=None)

 一种通用的数据加载方式，当输入以如下的格式存放时：
 root/class_a/1.ext
 root/class_a/2.ext
 root/class_a/3.ext

 root/class_b/123.ext
 root/class_b/456.ext
 root/class_b/789.ext

参数：
  - **root** (str) - 根目录路径。
  - **loader** (callable，可选) - 可以加载数据路径的一个函数，如果该值没有设定，默认使用 ``cv2.imread``  。默认值：None。
  - **extensions** (tuple[str]，可选) - 允许的数据后缀列表，如果该值没有设定，默认使用 ``('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')`` 。默认值：None。
  - **transform** (callable，可选) - 图片数据的预处理，若为 ``None`` 即为不做预处理。默认值为``None``。
  - **is_valid_file** (callable，可选) - 根据每条数据的路径来判断是否合法的一个函数。默认值：None。


**代码示例**：

.. code-block:: python

    import os
    import cv2
    import tempfile
    import shutil
    import numpy as np
    from paddle.vision.datasets import DatasetFolder

    def make_fake_dir():
        data_dir = tempfile.mkdtemp()

        for i in range(2):
            sub_dir = os.path.join(data_dir, 'class_' + str(i))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            for j in range(2):
                fake_img = (np.random.random((32, 32, 3)) * 255).astype('uint8')
                cv2.imwrite(os.path.join(sub_dir, str(j) + '.jpg'), fake_img)
        return data_dir

    temp_dir = make_fake_dir()
    # temp_dir is root dir
    # temp_dir/class_1/img1_1.jpg
    # temp_dir/class_2/img2_1.jpg
    data_folder = DatasetFolder(temp_dir)

    for items in data_folder:
        break
        
    shutil.rmtree(temp_dir)
