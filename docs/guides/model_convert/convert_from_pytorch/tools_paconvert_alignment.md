# PaConvert 单测验证与文档对齐工具

代码自动转换工具的开发可以分为两部分，即[撰写映射文档](./api_difference/pytorch_api_mapping_format_cn.html)和[**配置转换规则**](https://github.com/PaddlePaddle/PaConvert/blob/master/docs/CONTRIBUTING.md)。**配置转换规则**的产出主要包括 [api_mapping.json](https://github.com/PaddlePaddle/PaConvert/blob/master/paconvert/api_mapping.json)、[Matcher](https://github.com/PaddlePaddle/PaConvert/blob/master/paconvert/api_matcher.py) 和对应的 [单元测试](https://github.com/PaddlePaddle/PaConvert/tree/master/tests)。

由于 PyTorch api 功能复杂多样，且 PyTorch 历史遗留因素导致 api 功能风格多变，致使 API 转换规则错综复杂，难以通过自动分析的方式进行转换规则的推理与维护，因而转换规则目前均为人工编写、检查与维护。但随着 API 转换规则规模增大，人工检查维护成本日益繁重，带来了大量非必要心智负担。考虑到 API 转换规则开发与维护过程存在公共部分，可以在映射文档数据的支持下，对这些公共部分进行检查与验证。

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

-
