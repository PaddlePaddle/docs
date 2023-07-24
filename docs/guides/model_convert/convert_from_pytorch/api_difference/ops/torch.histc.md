## [ 参数不一致 ]torch.histc

### [torch.histc](https://pytorch.org/docs/stable/generated/torch.histc.html#torch-histc)

```python
torch.histc(input, bins=100, min=0, max=0, *, out=None)
```

### [paddle.histogram](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/histogram_cn.html#histogram)

```python
paddle.histogram(input, bins=100, min=0, max=0, name=None)
```

其中 PyTorch 与 Paddle 的返回值类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                |
| ------- | ------------ | --------------------------------------------------- |
| input   | input        | 表示输入的 Tensor。                                  |
| bins    | bins         | 表示直方图直条的个数。                              |
| min     | min          | 表示范围的下边界。                                  |
| max     | max          | 表示范围的上边界。                                  |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要进行转写。 |
| 返回值     | 返回值           | 表示返回值，PyTorch 的返回值类型为 float32，Paddle 的返回值类型为 int64 ， 需要进行转写。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.histc(x, out=y)

# Paddle 写法
paddle.assign(paddle.histogram(x).astype('float32'), y)
```
