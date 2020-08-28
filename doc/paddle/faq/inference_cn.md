# 预测引擎



## 常见问题

##### Q: 模型保存成功，但加载失败

+ 问题描述

VGG模型，训练时候使用`fluid.io.save_inference_model`保存模型，预测的时候使用`fluid.io.load_inference_model`加载模型文件。保存的是我自己训练的 VGG 模型。保存没问题，加载的时候报错`paddle.fluid.core.EnforceNotMet: Cannot read more from file` ？

+ 问题解答

错误提示可能的原因如下，请检查。

1、 模型文件有损坏或缺失。

2、 模型参数和模型结构不匹配。

## 同时多模型问题

##### Q: 加载两个模型失败

+ 问题描述

infer时，当先后加载检测和分类两个网络时，分类网络的参数为什么未被load进去？

+ 问题解答

尝试两个模型在不同的scope里面infer，使用`with fluid.scope_guard(new_scope)`，另外定义模型前加上`with fluid.unique_name.guard()`解决。

##### Q: 同时使用两个模型报错

+ 问题描述

两个模型都load之后，用第一个模型的时候会报错？

+ 问题解答

由于用`load_inference_model`的时候会修改一些用户不可见的环境变量，所以执行后一个`load_inference_model`的时候会把前一个模型的环境变量覆盖，导致前一个模型不能用，或者说再用的时候就需要再加载一次。此时需要用如下代码保护一下，[参考详情](https://github.com/PaddlePaddle/Paddle/issues/16661)。

```
xxx_scope = fluid.core.Scope()
with fluid.scope_guard(xxx_scope):
    [...] = fluid.load_inference_model(...)
```

##### Q: 多线程预测失败

+ 问题描述

c++调用paddlepaddle多线程预测出core？

+ 问题解答

Paddle predict 库里没有多线程的实现，当上游服务并发时，需要用户起多个预测服务，[参考示例](http://paddlepaddle.org/documentation/docs/zh/1.3/advanced_usage/deploy/inference/native_infer.html)。
