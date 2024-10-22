## [ torch 参数更多 ]torch.distributions.lkj_cholesky.LKJCholesky

### [torch.distributions.lkj_cholesky.LKJCholesky](https://pytorch.org/docs/stable/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky)

```python
torch.distributions.lkj_cholesky.LKJCholesky(dim, concentration=1.0, validate_args=None)
```

### [paddle.distribution.LKJCholesky](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/LKJCholesky_cn.html)

```python
paddle.distribution.LKJCholesky(dim, concentration=1.0, sample_method='onion')
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射


| PyTorch       | PaddlePaddle  | 备注                                                                    |
| ------------- | ------------- | ----------------------------------------------------------------------- |
| dim           | dim           | 表示输入的参数。                                                        |
| concentration | concentration | 表示输入的参数。                                                        |
| validate_args | -             | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
| -             | sample_method | pytorch 无此参数，paddle 保持默认即可。                                 |
| 返回值          | 返回值          | pytorch 比 Paddle 多一个维度，需转写。                                  |

### 转写示例

#### 返回值

```python
# PyTorch 写法
concentration = torch.tensor([1.0])
y = torch.distributions.lkj_cholesky.LKJCholesky(dim=3, concentration=concentration).sample()

# Paddle 写法
concentration = paddle.to_tensor([1.0])
y = paddle.distribution.LKJCholesky(dim=3, concentration=concentration).sample()
y = paddle.unsqueeze(y, axis=0)
```
