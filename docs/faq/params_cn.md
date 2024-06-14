# 参数调整常见问题

##### 问题：如何将本地数据传入`paddle.nn.embedding`的参数矩阵中？

+ 答复：需将本地词典向量读取为 NumPy 数据格式，然后使用`paddle.nn.initializer.Assign`这个 API 初始化`paddle.nn.embedding`里的`param_attr`参数，即可实现加载用户自定义（或预训练）的 Embedding 向量。

------

##### 问题：如何实现网络层中多个 feature 间共享该层的向量权重？

+ 答复：你可以使用`paddle.ParamAttr`并设定一个 name 参数，然后再将这个类的对象传入网络层的`param_attr`参数中，即将所有网络层中`param_attr`参数里的`name`设置为同一个，即可实现共享向量权重。如使用 embedding 层时，可以设置`param_attr=paddle.ParamAttr(name="word_embedding")`，然后把`param_attr`传入 embedding 层中。

----------


##### 问题：使用 optimizer 或 ParamAttr 设置的正则化和学习率，二者什么差异？

+ 答复：ParamAttr 中定义的`regularizer`优先级更高。若 ParamAttr 中定义了`regularizer`，则忽略 Optimizer 中的`regularizer`；否则，则使用 Optimizer 中的`regularizer`。ParamAttr 中的学习率默认为 1.0，在对参数优化时，最终的学习率等于 optimizer 的学习率乘以 ParamAttr 的学习率。

----------

##### 问题：如何导出指定层的权重，如导出最后一层的*weights*和*bias*？

+ 答复：

1. 在动态图中，使用`paddle.save` API， 并将最后一层的`layer.state_dict()` 传入至 save 方法的 obj 参数即可， 然后使用`paddle.load` 方法加载对应层的参数值。详细可参考 API 文档[save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/save_cn.html#save) 和[load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/load_cn.html#load)。
2. 在静态图中，使用`paddle.static.save_vars`保存指定的 vars，然后使用`paddle.static.load_vars`加载对应层的参数值。具体示例请见 API 文档：[load_vars](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/io/load_vars_cn.html) 和 [save_vars](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/io/save_vars_cn.html) 。

----------

##### 问题：训练过程中如何固定网络和 Batch Normalization（BN）？

+ 答复：

1. 对于固定 BN：设置 `use_global_stats=True`，使用已加载的全局均值和方差：`global mean/variance`，具体内容可查看官网 API 文档[batch_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/batch_norm_cn.html#batch-norm)。

2. 对于固定网络层：如： stage1→ stage2 → stage3 ，设置 stage2 的输出，假设为*y*，设置 `y.stop_gradient=True`，那么， stage1→ stage2 整体都固定了，不再更新。

----------

##### 问题：训练的 step 在参数优化器中是如何变化的？

<img src="https://paddlepaddleimage.cdn.bcebos.com/faqimage%2F610cd445435e40e1b1d8a4944a7448c35d89ea33ab364ad8b6804b8dd947e88c.png" width = "400" height = "200" alt="图片名称" align=center />

* 答复：

  `step`表示的是经历了多少组 mini_batch，其统计方法为`exe.run`(对应 Program)运行的当前次数，即每运行一次`exe.run`，step 加 1。举例代码如下：

```python
# 执行下方代码后相当于 step 增加了 N x Epoch 总数
for epoch in range(epochs):
    # 执行下方代码后 step 相当于自增了 N
    for data in [mini_batch_1,2,3...N]:
        # 执行下方代码后 step += 1
        exe.run(data)
```

-----


##### 问题：如何修改全连接层参数，比如 weight，bias？

+ 答复：可以通过`param_attr`设置参数的属性，`paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(0.0, 0.02), learning_rate=2.0)`，如果`learning_rate`设置为 0，该层就不参与训练。也可以构造一个 numpy 数据，使用`paddle.nn.initializer.Assign`来给权重设置想要的值。


-----


##### 问题：如何进行梯度裁剪？

+ 答复：Paddle 的梯度裁剪方式需要在[Optimizer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#api)中进行设置，目前提供三种梯度裁剪方式，分别是[paddle.nn.ClipGradByValue](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/ClipGradByValue_cn.html)`（设定范围值裁剪）`、[paddle.nn.ClipGradByNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/ClipGradByNorm_cn.html)`（设定 L2 范数裁剪）`
、[paddle.nn.ClipGradByGlobalNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/ClipGradByGlobalNorm_cn.html)`（通过全局 L2 范数裁剪）`，需要先创建一个该类的实例对象，然后将其传入到优化器中，优化器会在更新参数前，对梯度进行裁剪。

注：该类接口在动态图、静态图下均会生效，是动静统一的。目前不支持其他方式的梯度裁剪。

```python
linear = paddle.nn.Linear(10, 10)
clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)  # 可以选择三种裁剪方式
sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
sdg.step()                                      # 更新参数前，会先对参数的梯度进行裁剪
```
[了解更多梯度裁剪知识](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/01_paddle2.0_introduction/basic_concept/gradient_clip_cn.html)


----------

##### 问题：如何在同一个优化器中定义不同参数的优化策略，比如 bias 的参数 weight_decay 的值为 0.0，非 bias 的参数 weight_decay 的值为 0.01？

+ 答复：
  1. [AdamW](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/AdamW_cn.html#adamw)的参数`apply_decay_param_fun`可以用来选择哪些参数使用 decay_weight 策略。
  2. 在创建`Param`的时候，可以通过设置[ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ParamAttr_cn.html#paramattr)的属性来控制参数的属性。

----------

##### 问题：paddle fluid 如何自定义优化器，自定义更新模型参数的规则？
 + 答复：
   1. 要定义全新优化器，自定义优化器中参数的更新规则，可以通过继承 fluid.Optimizer，重写_append_optimize_op 方法实现。不同优化器实现原理各不相同，一般流程是先获取 learning_rate，gradients 参数，可训练参数，以及该优化器自身特别需要的参数，然后实现更新参数的代码，最后返回更新后的参数。
    在实现更新参数代码时，可以选择直接调用[paddle 的 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)或者使用[自定义原生算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/index_cn.html)。在使用自定义原生算子时，要注意动态图与静态图调用方式有所区别：
    需要首先使用`framework.in_dygraph_mode()`判断是否为动态图模式，如果是动态图模式，则需要调用`paddle._C_ops`中相应的优化器算子；如果不是动态图模式，则需要调用`block.append_op` 来添加优化器算子。
    代码样例可参考[paddle 源码](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/optimizer/optimizer.py)中 AdamOptimizer 等优化器的实现。
    2. 使用现有的常用优化器，可以在创建`Param`的时候，可以通过设置[ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ParamAttr_cn.html#paramattr)的属性来控制参数的属性，可以通过设置`regularizer`，`learning_rate`等参数简单设置参数的更新规则。
