# PaConvert 单测验证与文档对齐工具

代码自动转换工具的开发可以分为两部分，即[撰写映射文档](./api_difference/pytorch_api_mapping_format_cn.html)和[**配置转换规则**](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md)。**配置转换规则**的产出主要包括 [api_mapping.json](https://github.com/PaddlePaddle/PaConvert/blob/master/paconvert/api_mapping.json)、[Matcher](https://github.com/PaddlePaddle/PaConvert/blob/master/paconvert/api_matcher.py) 和对应的 [单元测试](https://github.com/PaddlePaddle/PaConvert/tree/master/tests)。

由于 PyTorch api 功能复杂多样，且 PyTorch 历史遗留因素导致 api 功能风格多变，致使 API 转换规则错综复杂，难以通过自动分析的方式进行转换规则的推理与维护，因而转换规则目前均为人工编写、检查与维护。但随着 API 转换规则规模增大，人工检查维护成本日益繁重，带来了大量非必要心智负担。考虑到 API 转换规则开发与维护过程存在公共部分，可以在映射文档数据与 `api_mapping.json` 的支持下，对这些公共部分进行检查与验证。

## 快速使用

验证单个单测文件：

```bash
python tools/validate_unittest/validate_unittest.py -r unittest_file_path
```

验证结果会写入 `tools/validate_unittest/validation_report.md` 文件，检查所验证单测的 api 是否在表格中，即可判断其是否符合规范，或不符合哪几种规范。

与映射文档对齐：

```bash
python tools/validate_docs/validate_docs.py --docs_mappings docs_mapping_file_path
```

其中 `docs_mapping_file_path` 为映射文档验证工具得到的 `docs_mappings.json` 文件，如果不指定该参数，则默认搜索 `tools/validate_docs/` 目录是否有 `docs_mappings.json`。

对齐结果包括：

- 已有 `Matcher` 的 `torch api` 是否有映射文档；
- `torch api` 对应的 `paddle api` 在 docs 和 PaConvert 中是否一致；
- `torch api` 对应的 docs 中的 **参数映射表** 与 PaConvert 中的 `kwargs_change` 是否一致；
- `torch api` 对应的 docs 中的函数签名与 PaConvert 中的 `args_list` 是否一致；


## 单元测试验证工具 validate_unittest

根据开发规范，单元测试需要拥有满足以下四种情况的测试用例：
- 所有关键字不指定
- 所有关键字都指定
- 关键字乱序
- 不指定所有默认参数

基于该需求，开发单元测试验证工具，负责以下两方面功能：

1. 基于所有单测执行结果，进行单测的多样性检查（是否覆盖四种情况）；
2. 对于单个单测，尝试进行自动补全缺乏的用例覆盖情况。

### 单测多样性验证

在首次使用单测验证工具或更新部分单测数据时，需要收集对应的单测数据。

在 `-r` 参数指定要重新收集的目录或单测文件路径，通过设置 pytest 插件 `RecordAPIRunPlugin`，工具使用 pytest 框架对指定的路径进行单测的发现、收集，从而得到每个 api 执行的单测代码。

*注意：即使单测出错，也可以正常收集数据。等待全部执行完即可，会生成 `tools/validate_unittest/validation.json` 数据文件*

在 `check_call_variety` 函数中，对于 `torch_api` 对应的单测代码 `code` ，找到 `torch_api` 在 `code` 中的首次出现位置，检查其调用的参数列表，并对其进行内容分析，与 `api_mapping.json` 的 `args_list`、`min_input_args` 进行对照，进行四种判断：
- 全部不指定关键字 `all args`
- 全部指定关键字 `all kwargs`
- 改变关键字顺序 `kwargs out of order`
- 全部不指定默认值 `all default`

此时默认会生成 `tools/validate_unittest/validation_report.md` 作为多样性检测报告，其中仅包含多样性不符合的 api。

当 api 是可重载的、存在 corner case 等情况时，会在验证报告中进行特别标记。

如果添加参数 `--richtext`，则生成的检测报告中，api 携带其 torch 官网文档的链接。

在判断后，生成的单测验证报告将输出到 `tools/validate_unittest/validation_report.md` 中。

### 单测自动补全

现有单测不符合规范的案例中，大多数情况能满足其中的部分需求，如存在全部不指定关键字等。基于 `args_list` 和 `min_input_args`，能够基于满足的需求，推理得到缺少的用例。

***单测自动补全仅作为辅助工具，不保障可靠性和正确性，因此只支持每次调用补全一个 api。***

修复过程在 `autofix_single_api` 方法进行实现，对于修复后的单测用例需要逐个检查以确保正确性。

要使用自动修复功能，需要至少存在 `位置参数+关键字参数` 的数量等于 `args_list` 中总参数量的用例。

### 使用方法

#### 生成测试数据

当修改后需要**重新检测**时，用法和 `pytest` 基本一致，支持三种更新方式：

1. 全局重新生成

    ```bash
    python tools/validate_unittest/validate_unittest.py -r tests
    ```

2. 重新生成单个单测的数据

    ```bash
    python tools/validate_unittest/validate_unittest.py -r tests/test_Tensor_amax.py
    ```

3. 重新生成若干个单测的数据

    ```bash
    # 通配符
    python tools/validate_unittest/validate_unittest.py -r tests/test_Tensor_div*
    # 列表
    python tools/validate_unittest/validate_unittest.py -r tests/test_Tensor_divide.py tests/test_Tensor_div.py
    ```

#### 自动补全

对单个单测尝试进行自动补全（需要存在使用全部参数的用例）

```bash
python tools/validate_unittest/validate_unittest.py --autofix -r tests/test_Tensor_amax.py
```

## 文档对齐工具 validate_docs

根据[文档验证工具](./tools_doc_generate.html) 的结果 `docs_mappings.json` 与 `api_mapping.json` 数据，可以对文档与转换规则的匹配程度进行验证，从而避免低级错误。

对齐工具用法：

```bash
python tools/validate_docs/validate_docs.py --docs_mappings docs_mapping_file_path
```

其中 `docs_mapping_file_path` 为映射文档验证工具得到的 `docs_mappings.json` 文件，如果不指定该参数，则默认搜索 `tools/validate_docs/` 目录是否有 `docs_mappings.json`，可以从 docs 中复制或链接。

对齐结果包括：

- 已有 `Matcher` 的 `torch api` 是否有映射文档；
- `torch api` 对应的 `paddle api` 在 docs 和 PaConvert 中是否一致；
- `torch api` 对应的 docs 中的 **参数映射表** 与 PaConvert 中的 `kwargs_change` 是否一致；
- `torch api` 对应的 docs 中的函数签名与 PaConvert 中的 `args_list` 是否一致；

对于 `kwargs_change` 和 `args_list` 的检查过程如下：
1. 从文档数据中使用 `get_kwargs_mapping_from_doc` 提取 `kwargs_change` 信息 。
2. 对于特殊 `Matcher` 蕴含 `kwargs_change` 的信息记录在 `PRESET_MATCHER_KWARGS_CHANGE_PAIRS`；
3. 检查是否每个文档中描述的 `kwargs_change` 都在 paconvert 中有实现；
4. 检查 paconvert 的 `args_list` 是否都在 docs 函数签名中；
5. 对于支持重载或存在 corner case 的 API，在对齐失败时特殊标记；

对齐结果输出：

- `tools/validate_docs/validate_error_list.log`：对齐失败的所有 api
- `tools/validate_docs/paconvert_data_error_list.log`：不合规范的 paconvert api
- `tools/validate_docs/missing_docs_list.log`：有 `Matcher` 但无文档的 api
