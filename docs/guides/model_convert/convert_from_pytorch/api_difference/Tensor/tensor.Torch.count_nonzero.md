## [ 仅 paddle 参数更多 ] torch.Tensor.count_nonzero

### [torch.count_nonzero](https://pytorch.org/docs/stable/generated/torch.count_nonzero.html?highlight=count_nonzero#torch.count_nonzero)

```python
torch.count_nonzero(input, dim=None)
```

### [paddle.count_nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/count_nonzero_cn.html)

```python
paddle.count_nonzero(x, axis=None, keepdim=False, name=None)
```

两者功能一致，仅 paddle 参数更多，具体如下：
### 参数映射
| PyTorch  | PaddlePaddle | 备注                                                  |
|----------|--------------| ----------------------------------------------------- |
| input    | x            |  输入的 Tensor，仅参数名不一致。               |
| dim      | axis         |  指定对 x 进行计算的轴，仅参数名不一致。               |
|          | keepdim      |  是否在输出 Tensor 中保留减小的维度，PyTorch 无此参数， Paddle 保持默认即可。               |
