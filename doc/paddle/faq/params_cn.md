# 参数调整常见问题

##### 问题：如何将本地数据传入`paddle.nn.embedding`的参数矩阵中？

+ 答复：需将本地词典向量读取为NumPy数据格式，然后使用`paddle.nn.initializer.Assign`这个API初始化`paddle.nn.embedding`里的`param_attr`参数，即可实现加载用户自定义（或预训练）的Embedding向量。

------

##### 问题：如何实现网络层中多个feature间共享该层的向量权重？

+ 答复：你可以使用`paddle.ParamAttr`并设定一个name参数，然后再将这个类的对象传入网络层的`param_attr`参数中，即将所有网络层中`param_attr`参数里的`name`设置为同一个，即可实现共享向量权重。如使用embedding层时，可以设置`param_attr=paddle.ParamAttr(name="word_embedding")`，然后把`param_attr`传入embedding层中。

----------


##### 问题：使用optimizer或ParamAttr设置的正则化和学习率，二者什么差异？

+ 答复：ParamAttr中定义的`regularizer`优先级更高。若ParamAttr中定义了`regularizer`，则忽略Optimizer中的`regularizer`；否则，则使用Optimizer中的`regularizer`。ParamAttr中的学习率默认为1.0，在对参数优化时，最终的学习率等于optimizer的学习率乘以ParamAttr的学习率。

----------

##### 问题：如何导出指定层的权重，如导出最后一层的*weights*和*bias*？

+ 答复：
1. 在静态图中，使用`paddle.static.save_vars`保存指定的vars，然后使用`paddle.static.load_vars`加载对应层的参数值。具体示例请见API文档：[load_vars](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/io/load_vars_cn.html) 和 [save_vars](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/io/save_vars_cn.html) 。

2. 在动态图中，使用`paddle.save` API， 并将最后一层的layer.state_dict() 传入至save方法的obj 参数即可， 然后使用`paddle.load` 方法加载对应层的参数值。详细可参考API文档[save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/framework/io/save_cn.html#save) 和[load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/framework/io/load_cn.html#load)。

----------

##### 问题：训练过程中如何固定网络和Batch Normalization（BN）？

+ 答复：

1. 对于固定BN：设置 `use_global_stats=True`，使用已加载的全局均值和方差：`global mean/variance`，具体内容可查看官网API文档[batch_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/batch_norm_cn.html)。

2. 对于固定网络层：如： stage1→ stage2 → stage3 ，设置stage2的输出，假设为*y*，设置 `y.stop_gradient=True`，那么， stage1→ stage2整体都固定了，不再更新。

----------

##### 问题：训练的step在参数优化器中是如何变化的？  

<img src="https://paddlepaddleimage.cdn.bcebos.com/faqimage%2F610cd445435e40e1b1d8a4944a7448c35d89ea33ab364ad8b6804b8dd947e88c.png" width = "400" height = "200" alt="图片名称" align=center />

* 答复：

  `step`表示的是经历了多少组mini_batch，其统计方法为`exe.run`(对应Program)运行的当前次数，即每运行一次`exe.run`，step加1。举例代码如下：

```text
# 执行下方代码后相当于step增加了N x Epoch总数
for epoch in range(epochs):
    # 执行下方代码后step相当于自增了N
    for data in [mini_batch_1,2,3...N]:
        # 执行下方代码后step += 1
        exe.run(data)
```

-----


##### 问题：如何修改全连接层参数，比如weight，bias？  

+ 答复：可以通过`param_attr`设置参数的属性，`paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(0.0, 0.02), learning_rate=2.0)`，如果`learning_rate`设置为0，该层就不参与训练。也可以构造一个numpy数据，使用`paddle.nn.initializer.Assign`来给权重设置想要的值。

----------
