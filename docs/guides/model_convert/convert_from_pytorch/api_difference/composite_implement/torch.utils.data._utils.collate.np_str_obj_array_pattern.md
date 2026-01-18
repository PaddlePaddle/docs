## [ 组合替代实现 ]torch.utils.data.\_utils.collate.np\_str\_obj\_array\_pattern
### [torch.utils.data.\_utils.collate.np\_str\_obj\_array\_pattern](https://github.com/pytorch/pytorch/blob/2f023bf7b962e69c0de01b89af197388d9df27cc/torch/utils/data/_utils/collate.py#L20)
```python
torch.utils.data._utils.collate.np_str_obj_array_pattern
```

此常量定义了一个正则表达式（RegEx），用来判断一个 NumPy 数组的元素类型（dtype）是否是“字符串”或“对象”类型。

### 转写示例
```python
# PyTorch 写法
pattern = torch.utils.data._utils.collate.np_str_obj_array_pattern

# Paddle 写法
pattern = re.compile(r'[SaUO]')
```
