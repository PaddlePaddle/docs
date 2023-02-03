## torch.mm
### [torch.mm](https://pytorch.org/docs/stable/generated/torch.mm.html?highlight=mm#torch.mm)
```python
torch.mm(input, mat2, *, out=None)
```
### [paddle.matmul](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/matmul_cn.html)
```python
paddle.matmul(x, y, transpose_x=False, transpose_y=False, name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x            | 表示输入的第一个 Tensor。               |
| other        | y            | 表示输入的第二个 Tensor。             |
| -           | transpose_x            | 表示相乘前是否转置 x，PyTorch 无此参数。               |
| -        | transpose_y            | 表示相乘前是否转置 y，PyTorch 无此参数。             |
| out          | -        | 表示输出的 Tensor，PaddlePaddle 无此参数。  |


### 功能差异

#### 计算方式
***PyTorch***：只支持$n × m$的矩阵与$m × p$的矩阵相乘。
***PaddlePaddle***：支持的输入与`torch.matmul`相同。
