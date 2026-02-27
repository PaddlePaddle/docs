## [ 仅参数名不一致 ]torch.nn.utils.spectral_norm
### [torch.nn.utils.spectral\_norm](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html#torch.nn.utils.spectral_norm)
```python
torch.nn.utils.spectral_norm(module,
                                name='weight',
                                n_power_iterations=1,
                                eps=1e-12,
                                dim=None)
```

### [paddle.nn.utils.spectral\_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/utils/spectral_norm_cn.html#paddle.nn.utils.spectral_norm)
```python
paddle.nn.utils.spectral_norm(layer,
                                name='weight',
                                n_power_iterations=1,
                                eps=1e-12,
                                dim=None)
```
两者功能一致，参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| module        | layer        | 要添加权重谱归一化的层，参数名不一致。                                  |
