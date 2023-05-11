## [ 参数不一致 ]torch.Tensor.svd

### [torch.Tensor.svd](https://pytorch.org/docs/stable/generated/torch.Tensor.svd.html#torch.Tensor.svd)

```python
torch.Tensor.svd(some=True, compute_uv=True)
```

### [paddle.linalg.svd](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/svd_cn.html#svd)
```python
paddle.linalg.svd(x, full_matrics=False, name=None)
```

输入参数不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -           | x         | 表示输入的 Tensor 。                                           |
| some           | -         | 是否计算完整的 U 和 V 矩阵， 默认为 `True`， Paddle 无此参数。修改 full_matrics 实现。                                          |
| compute_uv         | -       | 是否返回零填充的 U 和 V 矩阵， 默认为 `True`， Paddle 无此参数。暂无转写方式。                                    |
| -             | full_matrics         | 是否计算完整的 U 和 V 矩阵，默认为 `False`， Pytorch 无此参数，Paddle 应设为 not some。  |


### 转写示例
#### some 是否计算完整的 U 和 V 矩阵
```python
# Pytorch 写法
y = a.svd(some=False)

# Paddle 写法
y = paddle.linalg.svd(a, full_matrices=True)
```
