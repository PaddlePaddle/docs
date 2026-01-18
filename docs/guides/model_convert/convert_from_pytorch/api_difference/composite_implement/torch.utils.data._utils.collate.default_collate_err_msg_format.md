## [ 组合替代实现 ]torch.utils.data.\_utils.collate.default\_collate\_err\_msg\_format
### [torch.utils.data.\_utils.collate.default\_collate\_err\_msg\_format](https://github.com/pytorch/pytorch/blob/2f023bf7b962e69c0de01b89af197388d9df27cc/torch/utils/data/_utils/collate.py#L112)
```python
torch.utils.data._utils.collate.default_collate_err_msg_format
```

此常量定义了当 DataLoader 无法自动将某个数据类型打成 Batch 时，抛出的那个 TypeError 的报错信息的模版。Paddle 无此 API，可替换为常量。

### 转写示例
```python
# PyTorch 写法
torch.utils.data._utils.collate.default_collate_err_msg_format.format("type")

# Paddle 写法
'default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}'.format("type")
```
