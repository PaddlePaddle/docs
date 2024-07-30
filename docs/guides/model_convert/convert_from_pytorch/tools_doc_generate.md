# PyTorch-Paddle 映射文档自动化工具

代码自动转换工具的开发可以分为两部分，即[**撰写映射文档**](./api_difference/pytorch_api_mapping_format_cn.html)和[配置转换规则](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md)。**撰写映射文档**的产出包括大量映射文档与汇总 [映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)，分别对应 docs 仓库中的 [api_difference/](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 目录与 [pytorch_api_mapping_cn.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.md)。

由于 PyTorch api 功能复杂多样，且 PyTorch 历史遗留因素导致 api 功能风格多变，致使参数映射方式错综复杂，难以通过自动分析的方式进行映射方式推理与管理，因而映射文档目前均为人工撰写、检查与维护。但随着映射文档规模增大，人工检查成本日益繁重，带来了大量非必要心智负担。考虑到映射文档规范存在公共结构，通过自动读取与分析这些公共结构，可以批量得到映射文档的**元信息**，从而为映射文档的检查提供便利。

## 映射文档结构分析

首先以 `torch.Tensor.arctan2` 映射文档为例，介绍其公共结构。

````markdown
## [ 仅参数名不一致 ]torch.Tensor.arctan2

### [torch.Tensor.arctan2](https://pytorch.org/docs/stable/generated/torch.arctan2.html#torch.arctan2)

```python
torch.Tensor.arctan2(other)
```

### [paddle.Tensor.atan2](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/Tensor_en.html)

```python
paddle.Tensor.atan2(y, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                              |
| --------- | ------------ | --------------------------------- |
| other     | y            | 表示输入的 Tensor ，仅参数名不一致。 |
````

可以看到，其包含的部分按照标题行可以划分成**映射文档标题**、**torch API**、**paddle API**与**参数映射**四个部分，其中：

- **映射文档标题**: 包含 `映射类型` 与 `torch api`；
- **torch API**：包含 `torch api`、`torch url`（torch 文档链接）、`torch signature`（函数签名）三部分；
- **paddle API**：包含 `paddle api`、`paddle url`（paddle 文档链接）、`paddle signature`（函数签名）三部分；
- **参数映射**：包含一个表格，表格每行包含 `torch arg`、`paddle arg` 和其备注。

可以发现，这些公共结构中蕴含的部分信息是重复的，因此我们可以建立约束，对于重复的部分进行检查与验证。

## 映射文档验证工具

我们
