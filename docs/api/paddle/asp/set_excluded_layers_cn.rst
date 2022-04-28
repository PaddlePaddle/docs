.. _cn_api_asp_set_excluded_layers:

set_excluded_layers
-------------------------------

.. py:function:: paddle.asp.set_excluded_layers(param_names, main_program=None)


將給定的參數名稱排除在ASP的支援名單位，意即給定的參數將不參與ASP訓練流程。


参数
:::::::::
    - **param_names** (List of Str) - 一個包含要排除ASP支援的參數名稱列。
    - **main_program** (Program, 可選) - 包含 :attr:`param_names` 的Program，預設為None, 意即使用 `paddle.static.default_main_program()`。
                                         動態圖模式下不須設定，保持None即可。

代码示例
:::::::::
COPY-FROM: paddle.asp.set_excluded_layers
