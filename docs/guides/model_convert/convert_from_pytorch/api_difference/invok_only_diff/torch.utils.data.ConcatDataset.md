## [ 仅 API 调用方式不一致 ]torch.utils.data.ConcatDataset

### [torch.utils.data.ConcatDataset](https://pytorch.org/docs/stable/generated/torch.utils.data.ConcatDataset.html#torch.utils.data.ConcatDataset)

```python
torch.utils.data.ConcatDataset(datasets)
```

### [paddle.io.ConcatDataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/ConcatDataset_cn.html#paddle/io/ConcatDataset_cn#cn-api-paddle-io-ConcatDataset)

```python
paddle.io.ConcatDataset(datasets)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
dataset = torch.utils.data.ConcatDataset([RandomDataset(2), RandomDataset(2)])

# Paddle 写法
dataset = paddle.io.ConcatDataset([RandomDataset(2), RandomDataset(2)])

```
