## [ 仅参数名不一致 ]torch.unique_consecutive
### [torch.unique_consecutive](https://pytorch.org/docs/stable/generated/torch.unique_consecutive.html?highlight=unique_consecutive#torch.unique_consecutive)

```python
torch.unique_consecutive(input,
                         return_inverse=False,
                         return_counts=False,
                         dim=None)
```

### [paddle.unique_consecutive](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/unique_consecutive_cn.html#unique-consecutive)

```python
paddle.unique_consecutive(x,
                          return_inverse=False,
                          return_counts=False,
                          axis=None,
                          dtype='int64',
                          name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                   |
| dim           | axis         | 指定选取连续不重复元素的轴，仅参数名不一致。 |
| -             | dtype        | 用于设置 inverse 或者 counts 的类型，PyTorch 无此参数，Paddle 保持默认即可。 |
