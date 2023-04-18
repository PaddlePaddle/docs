## [仅参数默认值不一致]torch.Tensor.uniform_

### [torch.Tensor.uniform_](https://pytorch.org/docs/stable/generated/torch.Tensor.uniform_.html#torch-tensor-uniform)

```python
torch.Tensor.uniform_(from=0, to=1)
```

### [paddle.Tensor.uniform](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#uniform-min-1-0-max-1-0-seed-0-name-none)

```python
paddle.Tensor.uniform(min=- 1.0, max=1.0, seed=0, name=None)
```

两者功能一致,仅参数默认值不一致,具体如下：

### 参数映射

| PyTorch | PaddlePaddle |                             备注                             |
| ------- | :----------: | :----------------------------------------------------------: |
| from    |     min      |       表示生成元素的起始位置，与 Pytorch 默认值不同。        |
| to      |     max      |       表示生成元素的结束位置，与 Pytorch 默认值不同。        |
| -       |     seed     | 表示用于生成随机数的随机种子，PyTorch 无此参数，Paddle 保持默认即可。 |
