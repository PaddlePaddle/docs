.. _cn_api_paddle_audio_backends_set_backend:

set_backend
-------------------------------

.. py:function:: paddle.audio.backends.set_backend(backend_name: str)

设置处理语音 I/O 的后端。

参数
::::::::::::

    - **backend_name** (str) - 语音 I/O 后端名称，现支持 ``'wave_backend'`` ，如果安装了 paddleaudio >=1.0.2，则也支持 ``'soundfile'`` 。

返回
:::::::::
无

代码示例
:::::::::

COPY-FROM: paddle.audio.backends.set_backend
