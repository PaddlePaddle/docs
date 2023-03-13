## torch.Tensor.unique
### [torch.Tensor.unique](https://pytorch.org/docs/stable/generated/torch.Tensor.unique.html?highlight=unique#torch.Tensor.unique)

```python
torch.Tensor.unique(sorted=True, return_inverse=False, return_counts=False, dim=None)
```

### [paddle.Tensor.unique](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#unique-return-index-false-return-inverse-false-return-counts-false-axis-none-dtype-int64-name-none)

```python
paddle.Tensor.unique(return_index=False, return_inverse=False, return_counts=False, axis=None, dtype='int64', name=None)
```

两者功能一致，torch 参数更多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| sorted        | -            | 是否返回前进行排序                                      |
| return_inverse| return_inverse        | 是否返回输入 Tensor 的元素对应在独有元素中的索引        |
| return_counts | return_counts        | 是否返回每个独有元素在输入 Tensor 中的个数             |
| dim           | axis        | 选取的轴                                                  |
| -             | return_index| 是否返回独有元素在输入 Tensor 中的索引|

### 转写示例

```python
# torch 写法
torch.Tensor.unique(sorted=True, return_inverse=False, return_counts=False, dim=1)

# paddle 写法
paddle.Tensor.unique(return_index=False, return_inverse=False, return_counts=False, axis=1)
```
