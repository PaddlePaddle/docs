## [ 返回参数类型不一致 ]torch.Tensor.sort

### [torch.Tensor.sort](https://pytorch.org/docs/stable/generated/torch.Tensor.sort.html#torch-tensor-sort)

```python
torch.Tensor.sort(dim=-1, descending=False, stable=False)
```

### [paddle.Tensor.sort](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sort_cn.html#sort)

```python
paddle.Tensor.sort(axis=-1, descending=False, stable=False)
```

两者功能一致但返回参数类型不同，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis         | 指定对输入 Tensor 进行运算的轴。默认值为-1, 仅参数名不一致。 |
| descending    | descending   | 指定算法排序的方向, 参数完全一致。     |
| stable        | stable       | 是否使用稳定排序。  |
| 返回值        | 返回值        | 表示以(Tensor, LongTensor)输出的元组，含义是排序后的返回值和对应元素索引。Paddle 无此参数，若返回排序后的元素，需要转写；若需要返回元素和元素索引，需要结合 argsort 进行转写。      |

注：PyTorch 返回 (Tensor, LongTensor)，Paddle 返回 Tensor 。

### 转写示例

#### 返回值

```python
# 若要返回排序后的元素和元素索引，需要结合 argsort 进行转写
# PyTorch 写法
input.sort(input, -1, True)

# Paddle 写法
input.sort(input, -1, True), input.argsort(input, -1, True)
```
