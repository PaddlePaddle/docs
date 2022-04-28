.. _cn_api_asp_decorate:

decorate
-------------------------------

.. py:function:: paddle.asp.decorate(optimizer)


裝飾給定的Optimizer為OptimizerWithSparsityGuarantee物件，使其在神經網路的訓練過程中插入對應的稀疏操作。
如果是動態圖模式，ASP會在此階段對支援的參數創建對應的mask變數。如果是靜態圖模式，則會在Optimizer.minimize()階段才創建。


参数
:::::::::
    - **optimizer** (Optimizer) - 用於神經網路訓練的優化器。


代码示例
:::::::::
COPY-FROM: paddle.asp.decorate
