## [ 仅参数名不一致 ] torch.distributions.kl.register_kl

### [torch.distributions.kl.register_kl](https://pytorch.org/docs/stable/distributions.html?highlight=register_kl#torch.distributions.kl.register_kl)

```python
torch.distributions.kl.register_kl(type_p, type_q)
```

### [paddle.distribution.register_kl](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/register_kl_cn.html)

```python
paddle.distribution.register_kl(cls_p, cls_q)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| type_p           | cls_p           | 实例 p 的分布类型，继承于 Distribution 基类，仅参数名不一致。               |
| type_q           | cls_q           | 实例 q 的分布类型，继承于 Distribution 基类，仅参数名不一致。               |
