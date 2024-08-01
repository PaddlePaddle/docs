# PyTorch-Paddle 映射文档自动化工具

代码自动转换工具的开发可以分为两部分，即[**撰写映射文档**](./api_difference/pytorch_api_mapping_format_cn.html)和[配置转换规则](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md)。**撰写映射文档**的产出包括大量映射文档与汇总 [映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)，分别对应 docs 仓库中的 [api_difference/](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference) 目录与 [pytorch_api_mapping_cn.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.md)。

由于 PyTorch api 功能复杂多样，且 PyTorch 历史遗留因素导致 api 功能风格多变，致使参数映射方式错综复杂，难以通过自动分析的方式进行映射方式推理与管理，因而映射文档目前均为人工撰写、检查与维护。但随着映射文档规模增大，人工检查成本日益繁重，带来了大量非必要心智负担。考虑到映射文档规范存在公共结构，通过自动读取与分析这些公共结构，可以批量得到映射文档的**元信息**，从而为映射文档的检查提供便利。

## 快速使用

在完成映射文档撰写后，调用验证工具进行一次验证：

```bash
python docs/guides/model_convert/convert_from_pytorch/validate_mapping_in_api_difference.py
```

当映射文档内容存在问题时，验证工具会自动输出对应问题部分的内容。

当验证工具可以通过，但生成工具出错（如 CI 未通过）时，很可能是因为该 API 在表格中会被生成多次，请检查 CI 最后的输出内容或在本地进行生成工具调用，检查生成结果是否符合预期。

## 映射文档结构分析

以 `torch.Tensor.arctan2` 映射文档为例，介绍其公共结构。

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

## 映射文档自动化工具

映射文档自动化工具包含两部分，分别是[验证工具](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/validate_mapping_in_api_difference.py)与[生成工具](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/apply_reference_from_api_difference.py)。

### 映射文档验证工具

根据映射文档的结构，可以设计状态机 `ParserState` 逐行处理映射文档，从而提取每篇映射文档的元数据，提取过程 `get_meta_from_diff_file` 包含以下检查：

- `文件名` 和 **映射文档标题** 中的 `torch_api`、**torch API** 中的 `torch_api` 是否匹配；
- `torch API` 是否以 `torch.` 开头 *（第三方库限制以对应的第三方库名开头）*；
- `torch signature` 是否成功解析（属性 `torch.api_name` 或者函数签名 `torch.api_name(args)`）；
- `torch API` 中 `torch api` 和 `torch signature` 解析结果是否一致；
- `paddle signature` 是否成功解析；
- `paddle API` 中 `paddle api` 和 `paddle signature` 解析结果是否一致；
- `参数映射` 表格是否成功解析（表格每行 3 列）
- 状态机结束状态：
    - `无参数` 或 `组合替代实现`：允许无参数映射表
    - `组合替代实现` 或更靠后的映射类型：允许无 **paddle API** 部分
    - 其他类型：必须有所有结构

通过设计这一系列约束，可以检查映射文档中粗心疏漏导致的错误，从而降低检查成本；此外，通过将文档解析得到结构化的元数据，能为后续流程提供数据支持。

验证工具的使用方法为 `python docs/guides/model_convert/convert_from_pytorch/validate_mapping_in_api_difference.py`，其生成的内容均在脚本同目录下。

*为兼容第三方库的 api 映射语义，新版本将 `torch API` 修改为 `src API`，`paddle API` 修改为 `dst API`。*

### 映射表生成工具

除了每个 api 的映射文档，随着映射文档的增删修改，映射表 [pytorch_api_mapping_cn.md](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.md) 的维护也会对开发者造成额外的心智负担。为减轻工作量，在验证工具的基础上，采用预处理命令对映射表内容进行声明，在文档构建过程中根据映射文档元信息进行构建，从而实现生成的自动化。

#### 生成工具用法

映射表与映射文档的目录都按照包进行规划，因此可以首先设计以表为单位的预处理命令 `REFERENCE-MAPPING-TABLE`，语义上表示一个映射表格的引用。

**API 表引用格式** 为 ``REFERENCE-MAPPING-TABLE(prefix, max_depth=1)``，即首先传入一个 `prefix`，用于限制其包名，如 `torch.nn.functional.`，用于发现所有 `functional.` 的 api，随后可选传入一个 `max_depth`，表示其取成员的最大深度，该参数用于区分 `torch.XX` 和 `torch 其他类 API`。

除了支持自动生成表，另外两个功能分别是 API 别名引用与未实现 API 声明。

**API 别名引用格式** 为 `ALIAS-REFERENCE-ITEM(alias_name, api_name)`，但该项通常不需要手动来撰写。

要获得 API 别名引用表的预处理命令，首先将 `PaConvert` 仓库的 `paconvert/api_alias_mapping.json` 链接或复制到 `docs/guides/model_convert/convert_from_pytorch/api_alias_mapping.json` 位置，随后调用验证工具即可在同目录下生成 `alias_macro_lines.tmp.md`，其内容为所有 API 别名引用表的预处理命令。

API 别名表的生成逻辑与单个 API 项映射类似，实现于 `apply_reference_to_row_ex` 方法中，使用 `api_name` 对应的元数据进行生成。

**未实现 API 声明格式** 为 `NOT-IMPLEMENTED-ITEM(torch_api, torch_api_url)`，该项需要手动进行维护，因为仓库中不含该项的映射文档，因此在参数中包含其展示需要的信息。

在按照对应规则创建预处理命令后，通过直接调用 `python docs/guides/model_convert/convert_from_pytorch/apply_reference_from_api_difference.py` 即可进行生成，将 `pytorch_api_mapping_cn.md` 中的预处理命令进行展开。

#### 生成工具原理

生成工具对映射表文件的处理包括两次读取和一次写入。

生成工具读取时，当遇到符合预期的表格表头即进入准备读取的状态，随后跳过表格的分隔线，开始对预处理命令的读取状态，直到所在行不是预处理命令时回到普通状态。

由于该读取逻辑可复用，因此将这部分逻辑实现在验证工具的 `process_mapping_index` 方法，通过传入 `item_processer` 回调和 `context` 上下文来控制行为，使用 `IndexParserState` 状态集来控制读取状态。

两次读取中，第一次读取用于分析表格匹配条件，第二次读取进行实际的预处理命令替换。

第一次读取时使用 `reference_table_scanner` 方法作为回调，收集所有的 API 表引用项，记录其参数作为 API 分类的条件。随后在生成工具的 `get_c2a_dict` 方法中对所有条件按照优先 `prefix` 长度降序，次优 `max_depth` 升序的顺序进行排序，并对所有映射文件元数据按照条件进行匹配。

第二次读取时使用 `reference_mapping_item_processer` 方法作为回调，对于所有需要处理的表格行进行转换，将转换结果写回 `context` 的 `output` 项中。

完成读取后，检查是否有 API 重复出现，如果重复出现则输出重复出现的 API 名称和所在行，不写回源文件并进行 CI 报错，
