.. _cn_api_audio_backends_set_backend:

set_backend
-------------------------------

.. py:function:: paddle.audio.backends.set_backend(backend_name: str)

设置处理语音I/O的后端。

参数
::::::::::::

    - **backend_name** (str) - 语音I/O后端名称，现支持‘wave_backend’,如果安装了paddleaudio >=1.0.2,则也支持‘soundfile’。
  
返回
:::::::::

代码示例
:::::::::

COPY-FROM: paddle.audio.backends.set_backend