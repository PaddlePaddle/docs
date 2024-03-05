## [仅参数名不一致]torch.nn.functional.gaussian_nll_loss

### [torch.nn.functional.gaussian_nll_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.gaussian_nll_loss.html#torch.nn.functional.gaussian_nll_loss)

```python
torch.nn.functional.gaussian_nll_loss(input, target, var, full=False, eps=1e-06, reduction='mean')
```

### [paddle.nn.functional.gaussian_nll_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/gaussian_nll_loss_cn.html#gaussian-nll-loss)

```python
paddle.nn.functional.gaussian_nll_loss(input, label, variance, full=False, epsilon=1e-6, reduction='mean', name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                                              |
| --------- | ------------ | ----------------------------------------------------------------- |
| input     | input        | 输入 Tensor。                                                     |
| target    | label        | 输入 Tensor，仅参数名不一致。                                     |
| var       | variance     | 输入 Tensor，仅参数名不一致。                                     |
| full      | full         | 是否在损失计算中包括常数项。                                      |
| eps       | epsilon      | 用于限制 variance 的值，使其不会导致除 0 的出现，仅参数名不一致。 |
| reduction | reduction    | 指定应用于输出结果的计算方式。                                    |
