## [ 仅 API 调用方式不一致 ]torch.utils.data.default_collate

### [torch.utils.data.default_collate](https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate)

```python
torch.utils.data.default_collate(batch)
```

### [paddle.io.dataloader.collate.default_collate_fn](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/dataloader/collate/default_collate_fn_cn.html#paddle/io/dataloader/collate/default_collate_fn_cn#cn-api-paddle-io-dataloader-collate-default_collate_fn)

```python
paddle.io.dataloader.collate.default_collate_fn(batch)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.tensor(torch.utils.data.default_collate([0, 1, 2, 3]))

# Paddle 写法
result = paddle.tensor(paddle.io.dataloader.collate.default_collate_fn([0, 1, 2, 3]))
```
