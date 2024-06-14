## [ 返回参数类型不一致 ]torch.sort

### [torch.sort](https://pytorch.org/docs/stable/generated/torch.sort.html?highlight=sort#torch.sort)

```python
torch.sort(input,
           dim=-1,
           descending=False,
           stable=False,
           *,
           out=None)
```

### [paddle.sort](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sort_cn.html#paddle.sort)

```python
paddle.sort(x,
            axis=-1,
            descending=False,
            stable=False,
            name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，同时两个 api 的返回参数类型不同，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                           |
| dim           | axis         | 指定对输入 Tensor 进行运算的轴。默认值为-1, 仅参数名不一致。 |
| descending    |descending    | 指定算法排序的方向。如果设置为 True，算法按照降序排序。如果设置为 False 或者不设置，按照升序排序。默认值为 False，参数名相同。     |
| stable        | stable        | 使排序程序更稳定，保证等价元素的顺序得以保留。            |
| out           | -            | 表示以(Tensor, LongTensor)输出的元组，含义是排序后的返回值和对应元素索引。Paddle 无此参数，若返回排序后的元素，需要转写；若需要返回元素和元素索引，需要结合 argsort 进行转写。      |

注：PyTorch 返回 (Tensor, LongTensor)，Paddle 返回 Tensor 。

### 转写示例
#### out：指定输出
```python
# 若要返回排序后的元素和元素索引，需要结合 argsort 进行转写
# PyTorch 写法
torch.sort(input, -1, True, out = (y, indices))

# Paddle 写法
paddle.assign(paddle.sort(input, -1, True), y)
paddle.assign(paddle.argsort(input, -1, True), indices)
```
