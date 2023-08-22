## [仅 paddle 参数更多]torch.scatter

### [torch.scatter](https://pytorch.org/docs/2.0/generated/torch.scatter.html?highlight=torch+scatter#torch.scatter)

```python
torch.scatter(input,dim, index, src, reduce=None)
```

### [paddle.put_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/put_along_axis_cn.html#cn-api-paddle-tensor-put-along-axis)

```python
paddle.put_along_axis(arr,indices, values, axis, reduce="assign")

```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注    |
| ------- | ------------ | ------- |
| input     | arr         | 表示输入的 Tensor ，仅参数名不一致。 |
| dim     | axis         | 表示在哪一个维度 scatter ，仅参数名不一致。 |
| index   | indices        | 表示输入的索引张量，仅参数名不一致。 |
| src     | values        | 表示需要插入的值，仅参数名不一致。 |
| reduce       | reduce       | 归约操作类型 。 |
