## [ 参数不一致 ]torch.utils.data.default_collate
### [torch.utils.data.default_collate](https://pytorch.org/docs/stable/data.html?highlight=default_collate#torch.utils.data.default_collate)

```python
torch.utils.data.default_collate(batch)
```

### [paddle.io.dataloader.collate.default_collate_fn]()

```python
paddle.io.dataloader.collate.default_collate_fn(batch)
```

返回参数类型不一致，需要转写。具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| batch        | batch        | 输入的用于组 batch 的数据。                                    |
| 返回值        | 返回值        | 返回参数类型不一致，当 batch 的元素为 numpy.ndarray 或 number 时， PyTorch 默认返回 torch.tensor, Paddle 默认返回 numpy.ndarray。                                    |


### 转写示例
#### 当 batch 的元素为 numpy.ndarray 或 number 时
```python
# PyTorch 写法
y = torch.utils.data.default_collate(batch)

# Paddle 写法
y = paddle.to_tensor(paddle.io.dataloader.collate.default_collate_fn(batch))
```

#### 当 batch 的元素为字典且字典的 value 为 numpy.ndarray 或 number 时
```python
# PyTorch 写法
y = torch.utils.data.default_collate(batch)

# Paddle 写法
y = paddle.io.dataloader.collate.default_collate_fn(batch)
for k, v in y.items():
    y[k] = paddle.to_tensor(v)
```
