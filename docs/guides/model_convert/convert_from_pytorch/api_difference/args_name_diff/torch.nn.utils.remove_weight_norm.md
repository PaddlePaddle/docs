## [ 仅参数名不一致 ]torch.nn.utils.remove_weight_norm
### [torch.nn.utils.remove\_weight\_norm](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.remove_weight_norm.html#torch.nn.utils.remove_weight_norm)
```python
torch.nn.utils.remove_weight_norm(module,
                                    name='weight')
```

### [paddle.nn.utils.remove\_weight\_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/utils/remove_weight_norm_cn.html#paddle.nn.utils.remove_weight_norm)
```python
paddle.nn.utils.remove_weight_norm(layer,
                                    name='weight')
```
两者功能一致，参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| module        | layer        | 要添加权重归一化的层，参数名不一致。                                    |
