.. _cn_api_paddle_audio_datasets_ESC50:

ESC50
-------------------------------

.. py:class:: paddle.audio.datasets.ESC50(mode: str = 'train', split: int = 1, feat_type: str = 'raw', archive=None, **kwargs)


`ESC50 <http://dx.doi.org/10.1145/2733373.2806390>`_ 数据集的实现。

参数
:::::::::

  - **mode** (str，可选) - ``'train'`` 或 ``'dev'`` 模式两者之一，默认值为 ``'train'``。
  - **split** (int，可选) - 默认是 1，指定 dev 的文件夹。
  - **feat_type** (str，可选) - 默认是 raw，raw 是原始语音，支持 mfcc，spectrogram，melspectrogram，logmelspectrogram。指定从音频提取的语音特征。
  - **archive** (dict，可选) - 默认是 None，类中已经设置默认 archive，指定数据集的下载链接和 md5 值。

返回
:::::::::

:ref:`cn_api_paddle_io_Dataset`，ESC50 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.audio.datasets.ESC50
