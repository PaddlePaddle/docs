## [torch 参数更多]torch.distributions.geometric.Geometric

### [torch.distributions.geometric.Geometric](https://pytorch.org/docs/stable/distributions.html#torch.distributions.geometric.Geometric)

```python
torch.distributions.geometric.Geometric(probs=None, logits=None, validate_args=None)
```

### [paddle.distribution.Geometric](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Geometric_cn.html#geometric)

```python
paddle.distribution.Geometric(probs)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| probs         | probs        | 几何分布成功概率参数。                                                  |
| logits        | -            | 采样 1 的 log-odds，Paddle 无此参数，暂无转写方式。  |
| validate_args | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
