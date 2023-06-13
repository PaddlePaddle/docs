## [仅 paddle 参数更多]torch.Tensor.uniform_

### [torch.Tensor.uniform_](https://pytorch.org/docs/stable/generated/torch.Tensor.uniform_.html#torch-tensor-uniform)

```python
torch.Tensor.uniform_(from=0, to=1)
```

### [paddle.Tensor.uniform](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#uniform-min-1-0-max-1-0-seed-0-name-none)

```python
paddle.Tensor.uniform(min=- 1.0, max=1.0, seed=0, name=None)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle |                             备注                             |
| ------- | :----------: | :----------------------------------------------------------: |
| from    |     min      |           表示生成元素的起始位置，仅参数名不一致。           |
| to      |     max      |           表示生成元素的结束位置，仅参数名不一致。           |
| -       |     seed     | 表示用于生成随机数的随机种子，PyTorch 无此参数，Paddle 保持默认即可。 |
