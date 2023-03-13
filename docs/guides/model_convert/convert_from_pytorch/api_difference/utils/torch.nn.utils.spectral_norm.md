## torch.nn.utils.spectral_norm
### [torch.nn.utils.spectral_norm](https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html?highlight=nn+utils+spectral_norm#torch.nn.utils.spectral_norm)

```python
torch.nn.utils.spectral_norm(module,
                                name='weight',
                                n_power_iterations=1,
                                eps=1e-12,
                                dim=None)
```

### [paddle.nn.utils.spectral_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/utils/spectral_norm_cn.html#spectral-norm)

```python
paddle.nn.utils.spectral_norm(layer,
                                name='weight',
                                n_power_iterations=1,
                                eps=1e-12,
                                dim=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| module        | layer        | 要添加权重谱归一化的层。                                  |
