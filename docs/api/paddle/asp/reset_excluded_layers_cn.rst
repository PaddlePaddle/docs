.. _cn_api_asp_reset_excluded_layers:

reset_excluded_layers
-------------------------------

.. py:function:: paddle.asp.reset_excluded_layers(main_program=None)

將被設定於排除在ASP訓練流程外的參數列表清空，意即所以可以被ASP支援的參數都將參與ASP訓練流程。


参数
:::::::::
    - **main_program** (Program, 可選) - 包含神經網絡參數的Program，預設為None, 意即使用 `paddle.static.default_main_program()`。
                                         動態圖模式下不須設定，保持None即可。

代码示例
:::::::::
COPY-FROM: paddle.asp.reset_excluded_layers
