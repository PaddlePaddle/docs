## [ 仅 API 调用方式不一致 ]torch.utils.data.ChainDataset

### [torch.utils.data.ChainDataset](https://pytorch.org/docs/stable/generated/torch.utils.data.ChainDataset.html#torch.utils.data.ChainDataset)

```python
torch.utils.data.ChainDataset(datasets)
```

### [paddle.io.ChainDataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/ChainDataset_cn.html#paddle/io/ChainDataset_cn#cn-api-paddle-io-ChainDataset)

```python
paddle.io.ChainDataset(datasets)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
dataset = torch.utils.data.ChainDataset(
    [MyIterableDataset(start=3, end=7), MyIterableDataset(start=3, end=7)]
)

# Paddle 写法
dataset = paddle.io.ChainDataset(
    [MyIterableDataset(start=3, end=7), MyIterableDataset(start=3, end=7)]
)
```
