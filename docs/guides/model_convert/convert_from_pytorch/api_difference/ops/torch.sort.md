## [torch 参数更多 ]torch.sort

### [torch.sort](https://pytorch.org/docs/stable/generated/torch.sort.html?highlight=sort#torch.sort)

```python
torch.sort(input,
           dim=- 1,
           descending=False,
           stable=False,
           *,
           out=None)
```

### [paddle.sort](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sort_cn.html#paddle.sort)

```python
paddle.sort(x,
            axis=- 1,
            descending=False,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不同。                          |
| dim           | axis         | 指定对输入 Tensor 进行运算的轴。默认值为-1, 仅参数名不同。|
| descending    |descending    | 指定算法排序的方向。如果设置为 True，算法按照降序排序。如果设置为 False 或者不设置，按照升序排序。默认值为 False，参数名相同。     |
| out           | -            | 表示以(Tensor, LongTensor)输出的元组 ，需要进行转写      |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.sort([[4, 1, 7], [1, 6, 3], [9, 4, 11]], 0, True, (y, l_y))

# Paddle 写法
paddle.assign(paddle.sort([[4, 1, 7], [1, 6, 3], [9, 4, 11]], 0, True), y)
```
