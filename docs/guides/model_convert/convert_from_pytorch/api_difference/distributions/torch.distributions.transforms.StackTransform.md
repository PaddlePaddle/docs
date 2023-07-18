## [torch 参数更多]torch.distributions.transforms.StackTransform

### [torch.distributions.transforms.StackTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.StackTransform)

```python
torch.distributions.transforms.StackTransform(tseq, dim=0, cache_size=0)
```

### [paddle.distribution.StackTransform](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/StackTransform_cn.html)

```python
paddle.distribution.StackTransform(transforms, axis=0)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                                                       |
| ---------- | ------------ | -------------------------------------------------------------------------- |
| tseq       | transforms   | 输入的变换序列，仅参数名不一致。                                           |
| dim        | axis         | 待变换的轴，仅参数名不一致。                                               |
| cache_size | -            | 表示 cache 大小，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
