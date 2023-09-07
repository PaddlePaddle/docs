.. _cn_api_paddle_audio_functional_get_window:

get_window
-------------------------------

.. py:function:: paddle.audio.functional.get_window(window, win_length, fftbins=True, dtype='float64')

根据参数给出对应长度和类型的窗函数。

参数
::::::::::::

    - **window** (str 或者 Tuple[str，float]) - 窗函数类型，或者(窗参数类型， 窗函数参数)，支持的窗函数类型'hamming'，'hann'，'gaussian'，'general_gaussian'，'exponential'，'triang'，'bohman'，'blackman'，'cosine'，'tukey'，'taylor'。
    - **win_length** (int) - 采样点数。
    - **fftbins** (bool，可选) -  如果是 True，给出一个周期性的窗，如果是 False 给出一个对称性的窗，默认是 True。
    - **dtype** (str，可选) - 默认'float64'。

返回
:::::::::

``paddle.Tensor``，对应窗表征的 Tensor 。

代码示例
:::::::::

COPY-FROM: paddle.audio.functional.get_window
