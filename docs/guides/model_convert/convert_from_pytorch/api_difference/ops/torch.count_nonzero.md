## [仅参数名不一致]torch.count_nonzero
### [torch.count_nonzero](https://pytorch.org/docs/stable/generated/torch.count_nonzero.html?highlight=count_nonzero#torch.count_nonzero)

```python
torch.count_nonzero(input, dim=None)
```

### [paddle.count_nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/count_nonzero_cn.html#count-nonzero)

```python
paddle.count_nonzero(x,
                    axis=None,
                    keepdim=False,
                    name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示进行运算的轴，仅参数名不一致。  |
| - | <font color='red'> keepdim </font> | 是否在输出 Tensor 中保留减小的维度， Pytorch 无此参数， Paddle 保持默认即可。  |
