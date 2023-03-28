## [ 仅 Paddle 参数更多 ] torch.Tensor.count_nonzero

### [torch.Tensor.count_nonzero](https://pytorch.org/docs/stable/generated/torch.Tensor.count_nonzero.html?highlight=count_non#torch.Tensor.count_nonzero)

```python
torch.Tensor.count_nonzero(dim=None)
```

### [paddle.Tensor.count_nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#count-nonzero-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.count_nonzero(axis=None, keepdim=False, name=None)
```

两者功能一致，仅 paddle 参数更多，具体如下：
### 参数映射
| PyTorch  | PaddlePaddle | 备注                                                  |
|----------|--------------| ----------------------------------------------------- |
| dim      | axis         |  指定对 x 进行计算的轴，仅参数名不一致。               |
|          | keepdim      |  是否在输出 Tensor 中保留减小的维度，PyTorch 无此参数， Paddle 保持默认即可。               |
