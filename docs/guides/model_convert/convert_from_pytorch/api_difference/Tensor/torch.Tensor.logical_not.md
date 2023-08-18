## [ 仅 paddle 参数更多 ] torch.Tensor.logical_not

### [torch.Tensor.logical_not](https://pytorch.org/docs/stable/generated/torch.Tensor.logical_not.html)

```python
torch.Tensor.logical_not()
```

### [paddle.Tensor.logical_not](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#logical-not-out-none-name-none)

```python
paddle.Tensor.logical_not(out=None,
                          name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                               |
| ------- | ------------ | ---------------------------------- |
| -   | out            | 指定算子输出结果的 Tensor，可以是程序中已经创建的任何 Tensor。默认值为 None，此时将创建新的 Tensor 来保存输出结果。Pytorch 无此参数，Paddle 保持默认即可。|
