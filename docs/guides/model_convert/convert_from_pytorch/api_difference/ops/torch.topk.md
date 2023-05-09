## [ 参数不一致 ]torch.topk

### [torch.topk](https://pytorch.org/docs/stable/generated/torch.topk.html?highlight=topk#torch.topk)

```python
torch.topk(input,
           k,
           dim=None,
           largest=True,
           sorted=True,
           *,
           out=None)
```

### [paddle.topk](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/topk_cn.html#paddle.topk)

```python
paddle.topk(x,
            k,
            axis=None,
            largest=True,
            sorted=True,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，同时两个 api 的返回值不同，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不同。                          |
| k             | k            | 在指定的轴上进行 top 寻找的数量， 参数名相同。                          |
| dim           | axis         | 指定对输入 Tensor 进行运算的轴。默认值为-1, 仅参数名不同。|
| largest       | largest      | 指定算法排序的方向。如果设置为 True，算法按照降序排序，否则按照升序排序。默认值为 True，参数名相同。     |
| sorted        | sorted       | 控制返回的结果是否按照有序返回，默认为 True。在 GPU 上总是返回有序的结果。参数名相同。 |
| out           | -            | 表示以(Tensor, LongTensor)输出的元组，含义是查找topk后的返回值和对应元素索引。Paddle 无此参数，需要进行转写。  |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.topk(input, k=1, dim=-1, out = (y, indices))

# Paddle 写法
paddle.assign(paddle.topk(input, k=1, axis=-1), (y, indices))
```
