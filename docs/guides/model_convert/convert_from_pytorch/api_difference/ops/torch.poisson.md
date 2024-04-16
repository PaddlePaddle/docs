## [ torch 参数更多 ]torch.poisson

### [torch.poisson](https://pytorch.org/docs/stable/generated/torch.poisson.html#torch.poisson)
```python
torch.poisson(input,
              generator=None)
```
### [paddle.poisson](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/poisson_cn.html)
```python
paddle.poisson(x,
               name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input              |  x           | 表示输入的 Tensor ，仅参数名不一致。  |
| generator           |  -           | 用于采样的伪随机数生成器，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
