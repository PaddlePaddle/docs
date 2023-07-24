## [torch 参数更多]torch.Tensor.exponential\_

### [torch.Tensor.exponential\_](https://pytorch.org/docs/stable/generated/torch.Tensor.exponential_.html#torch.Tensor.exponential_)

```python
torch.Tensor.exponential_(lambd=1, *, generator=None)
```

### [paddle.Tensor.exponential\_](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#exponential-lam-1-0-name-none)

```python
paddle.Tensor.exponential_(lam=1.0, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                                                                |
| --------- | ------------ | ----------------------------------------------------------------------------------- |
| lambd     | lam          | 指数分布的 λ 参数，仅参数名不一致。                                                 |
| generator | -            | 用于采样的伪随机数生成器，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
