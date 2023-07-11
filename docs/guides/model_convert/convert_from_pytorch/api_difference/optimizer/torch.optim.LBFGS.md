## [ Paddle 参数更多 ]torch.optim.LBFGS

### [torch.optim.LBFGS](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html)

```python
torch.optim.LBFGS(params,
                lr=1,
                max_iter=20,
                max_eval=None,
                tolerance_grad=1e-07,
                tolerance_change=1e-09,
                history_size=100,
                line_search_fn=None)
```

### [paddle.optimizer.LBFGS](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/LBFGS_cn.html)

```python
paddle.optimizer.LBFGS(learning_rate=1.0,
                        max_iter=20,
                        max_eval=None,
                        tolerance_grad=1e-07,
                        tolerance_change=1e-09,
                        history_size=100,
                        line_search_fn=None,
                        parameters=None,
                        weight_decay=None,
                        grad_clip=None,
                        name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不同。                      |
| lr     | learning_rate       | 学习率，用于参数更新的计算。仅参数名不同。                          |
| max_iter   | max_iter   | 每个优化单步的最大迭代次数。参数名和默认值均一致。                       |
| max_eval       | max_eval     | 每次优化单步中函数计算的最大数量。参数名和默认值均一致。                           |
| tolerance_grad       | tolerance_grad    |  当梯度的范数小于该值时，终止迭代。参数名和默认值均一致。         |
| tolerance_change       | tolerance_change    |  当函数值/x 值/其他参数 两次迭代的改变量小于该值时，终止迭代。参数名和默认值均一致。         |
| history_size       | history_size    |  指定储存的向量对{si,yi}数量。参数名和默认值均一致。         |
| line_search_fn      | line_search_fn    |  指定要使用的线搜索方法。参数名和默认值均一致。         |
| -           | weight_decay     | 表示权重衰减系数。PyTorch 无此参数，Paddle 保持默认即可。         |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |
