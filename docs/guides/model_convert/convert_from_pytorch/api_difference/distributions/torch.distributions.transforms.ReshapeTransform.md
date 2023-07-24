## [torch 参数更多]torch.distributions.transforms.ReshapeTransform

### [torch.distributions.transforms.ReshapeTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.ReshapeTransform)

```python
torch.distributions.transforms.ReshapeTransform(in_shape, out_shape, cache_size=0)
```

### [paddle.distribution.ReshapeTransform](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/ReshapeTransform_cn.html)

```python
paddle.distribution.ReshapeTransform(in_event_shape, out_event_shape)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle    | 备注                                        |
| ---------- | --------------- | ------------------------------------------- |
| in_shape   | in_event_shape  | Reshape 前的事件形状，仅参数名不一致。      |
| out_shape  | out_event_shape | Reshape 后的事件形状，仅参数名不一致。      |
| cache_size | -               | cache 大小，Paddle 无此参数，暂无转写方式。 |
