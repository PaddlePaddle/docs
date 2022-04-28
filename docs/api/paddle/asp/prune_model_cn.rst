.. _cn_api_asp_prune_model:

prune_model
-------------------------------

.. py:function:: paddle.asp.prune_model(model, n=2, m=4, mask_algo='mask_1d', with_mask=True)


將 :attr:`model`裡被ASP支援的參數根據 :attr:`mask_algo` 裁減為 :attr:`n` : :attr:`m`的稀疏模式。此函式支援訓練與推理的裁減並透過
:attr:`with_mask` 來控制，若 :attr:`with_mask` 為True，表示啟用訓練裁減模式，意即將ASP masks相關的變數一同進行裁減。

*Note*: 在靜態模式下，如果設定 :attr:`with_mask`為True，則必須先呼叫 `OptimizerWithSparsityGuarantee.minimize` 
與初始化(`exe.run(startup_program`)

参数
:::::::::
    - **model** (Layer|Program) - 儲存神經網路參數的物件，可以式nn.Layer，或者靜態圖的static.Program。
    - **n** (int) - n:m 稀疏模式中的n。
    - **m** (int) - n:m 稀疏模式中的m。
    - **mask_algo** (str, 可選) - 用於產生稀疏模式的演算法，可為 `mask_1d`, `mask_2d_greedy` 與 `mask_2d_best`，預設為 `mask_1d`。
    - **with_mask** (bool, 可選) - 決定是否對ASP支援的參數所對應的mask變數進行裁減，預設為True。


代码示例
:::::::::
COPY-FROM: paddle.asp.prune_model
