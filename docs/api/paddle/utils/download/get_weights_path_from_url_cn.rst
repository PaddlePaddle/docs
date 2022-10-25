.. _cn_api_paddle_utils_download_get_weights_path_from_url:

get_weights_path_from_url
-------------------------------

.. py:function:: paddle.utils.download.get_weights_path_from_url(url, md5sum=None)

 从 ``WEIGHT_HOME`` 文件夹获取权重，如果不存在，就从 url 下载。

参数
::::::::::::

  - **url** (str) - 下载的链接。
  - **md5sum** (str，可选) - 下载文件的 md5 值。默认值：None。

返回
::::::::::::
权重的本地路径。


代码示例
::::::::::::

COPY-FROM: paddle.utils.download.get_weights_path_from_url
