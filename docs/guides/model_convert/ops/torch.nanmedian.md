## torch.nanmedian
### [torch.nanmedian](https://pytorch.org/docs/stable/generated/torch.nanmedian.html?highlight=nanmedian#torch.nanmedian)

```python
torch.nanmedian(input,
                dim=-1,
                keepdim=False,
                *,
                out=None)
```

### [paddle.nanmedian](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nanmedian_cn.html#nanmedian)

```python
paddle.nanmedian(x,
                 axis=None,
                 keepdim=True,
                 name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                   |
| dim           | axis         | 指定对 x 进行计算的轴。               |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。  |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.nanmedian(a, -1, out=y)

# Paddle 写法
y = paddle.nanmedian(a, -1)
```
