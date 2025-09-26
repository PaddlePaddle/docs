## [ 仅参数名不一致 ] torch.no_grad

### [torch.no_grad](https://pytorch.org/docs/stable/generated/torch.no_grad.html)

```python
torch.no_grad(orig_func=None)
```

### [paddle.no_grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/no_grad_cn.html)

```python
paddle.no_grad(func=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                      |
| ----------- | ------------ | ----------------------------------------------------------------------------------------- |
| orig_func   | func         | no_grad 装饰器所应用的对象，仅参数名不同。no_grad 作为上下文管理器使用时，可忽略该参数。       |
