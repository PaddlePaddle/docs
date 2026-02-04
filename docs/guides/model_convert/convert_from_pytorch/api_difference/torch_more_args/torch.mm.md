## [ torch 参数更多 ]torch.mm
### [torch.mm](https://docs.pytorch.org/docs/stable/generated/torch.mm.html#torch.mm)
```python
torch.mm(input,
         mat2,
         *,
         out=None)
```

### [paddle.mm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/mm_cn.html#paddle.mm)
```python
paddle.mm(input, mat2, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，Paddle 多余参数保持默认即可，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | input            | 表示输入的第一个 Tensor。               |
| mat2          | mat2            | 表示输入的第二个 Tensor。             |
|  out  | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.mm(a, b, out=y)

# Paddle 写法
paddle.assign(paddle.mm(a, b), y)
```
