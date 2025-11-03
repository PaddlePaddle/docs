## [ 仅 API 调用方式不一致 ]torch.utils.data.Subset

### [torch.utils.data.Subset](https://pytorch.org/docs/stable/generated/torch.utils.data.Subset.html#torch.utils.data.Subset)

```python
torch.utils.data.Subset(dataset, indices)
```

### [paddle.io.Subset](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/Subset_cn.html#paddle/io/Subset_cn#cn-api-paddle-io-Subset)

```python
paddle.io.Subset(dataset, indices)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
subset = torch.utils.data.Subset(full_dataset, indices)

# Paddle 写法
subset = paddle.io.Subset(full_dataset, indices)

```
