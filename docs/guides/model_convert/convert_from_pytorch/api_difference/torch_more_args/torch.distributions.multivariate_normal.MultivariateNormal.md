## [ torch 参数更多 ]torch.distributions.multivariate_normal.MultivariateNormal
### [torch.distributions.multivariate\_normal.MultivariateNormal](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal)
```python
torch.distributions.multivariate_normal.MultivariateNormal(loc,
                                       covariance_matrix=None,
                                       precision_matrix=None,
                                       scale_tril=None,
                                       validate_args=None)
```

### [paddle.distribution.MultivariateNormal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/MultivariateNormal_cn.html#paddle.distribution.MultivariateNormal)
```python
paddle.distribution.MultivariateNormal(loc,
                                       covariance_matrix=None,
                                       precision_matrix=None,
                                       scale_tril=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| ------------- | ------ | ------------------------------------------------------------ |
| loc           | loc      |  MultivariateNormal 的均值向量。         |
| covariance_matrix           | covariance_matrix      | MultivariateNormal 的协方差矩阵。         |
| precision_matrix        | precision_matrix      | MultivariateNormal 协方差矩阵的逆矩阵。 |
| scale_tril        | scale_tril      | MultivariateNormal 协方差矩阵的柯列斯基分解的下三角矩阵。 |
| validate_args        | -      | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
