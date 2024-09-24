## [torch 参数更多 ]torch.distributions.studentT.StudentT

### [torch.distributions.studentT.StudentT](https://pytorch.org/docs/stable/distributions.html#studentt)

```python
torch.distributions.studentT.StudentT(df, loc=0.0, scale=1.0, validate_args=None)
```

### [paddle.distribution.StudentT](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/StudentT_cn.html)

```python
paddle.distribution.StudentT(df, loc, scale, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle  | 备注                                                                    |
| ------------- | ------------- | ----------------------------------------------------------------------- |
| df           | df           | 表示输入的参数。                                                        |
| loc | loc | 表示输入的参数。                                                        |
| scale | scale | 表示输入的参数。                                                        |
| validate_args | -             | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
