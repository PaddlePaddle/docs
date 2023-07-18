## [参数完全一致]torch.utils.data.Subset

### [torch.utils.data.Subset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset)

```python
torch.utils.data.Subset(dataset, indices)
```

### [paddle.io.Subset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/Subset_cn.html)

```python
paddle.io.Subset(dataset, indices)
```

paddle 参数和 torch 参数完全一致，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                  |
| ---------- | ----------- | ------------------------------------- |
| dataset | dataset  | 原数据集 |
| indices | indices  | 用于提取子集的原数据集合指标数组 |
