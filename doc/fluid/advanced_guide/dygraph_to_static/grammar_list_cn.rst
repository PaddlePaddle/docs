ProgramTranslator支持的语法
==========================

由于ProgramTranslator本质是把python运行语法转写为PaddlePaddle静态图代码，但是python语法的表达能力和PaddlePaddle静态图表达能力肯定存在不同。使得一些代码无法被转换。

本章节我们将详细讲述在转换过程我们转化了哪些语法，哪些是无法转化的语法情况，并且推荐用户怎么改写这种语法使之能被支持。

我们将支持的语法分为以下几个大类

控制流相关关键词
------------------

控制流指if-elif-else，while等能够控制程序语句执行顺序的关键字。PaddlePaddle静态图具有cond，while_loop API来实现条件判断和循环，在动转静过程中这些关键字的转化情况为：

1. if-elif-else 条件

当if <条件> 中的条件是Tensor时，ProgramTranslator会把该if-elif-else语句转化为等价的cond

2. 

