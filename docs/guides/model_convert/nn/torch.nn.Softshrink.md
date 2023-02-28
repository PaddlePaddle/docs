## torch.nn.Softshrink
### [torch.nn.Softshrink](https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html?highlight=nn+softshrink#torch.nn.Softshrink)

```python
torch.nn.Softshrink(lambd=0.5)
```

### [paddle.nn.Softshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Softshrink_cn.html#softshrink)

```python
paddle.nn.Softshrink(threshold=0.5,
                        name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| lambd         | threshold    | Softshrink 激活计算公式中的阈值，必须大于等于零。            |
