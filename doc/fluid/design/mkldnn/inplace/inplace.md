## Introduction

PaddlePaddle is implementing concept of in-place execution of some of operators. Currently assumption is that
if operator can have in-place processing then all its kernel (including oneDNN) should be able to work
properly in in-place mode. To match this functionality oneDNN integration was extended to support in-place execution for 
some of its operators:
- activations
- softmax
- elementwise_add
- gelu*
- sum**

![](images/inplace.svg)   

There are two parts of in-place support:
- in-place execution support within an operator
- onednn inplace C-API pass 

* onednn gelu kernel is able to perform in-place execution , but currently gelu op does not support in-place support
