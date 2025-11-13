# PyTorch-Paddle 映射文档自动化工具

本工具链旨在自动化生成与校验 PyTorch 与 PaddlePaddle 的 API 映射文档，通过解析差异文档、生成结构化数据、校验格式与一致性，确保映射文档的准确性与统一性。

---

## 目录结构
```bash
tools/
├── get_api_difference_info.py        # 解析差异文档并生成结构化数据
├── generate_pytorch_api_mapping.py   # 生成最终映射文档
├── validate_api_difference_consistency.py  # 校验差异文档与 PaConvert 内容一致性
├── validate_api_difference_format.py # 校验差异文档格式
├── validate_pytorch_api_mapping.py   # 校验映射文档格式与内容
└── utils.py                          # 工具函数库
```

---

## 文件说明

### 1. `get_api_difference_info.py`
**功能**:
- 使用状态机 `ParserState` 逐行解析差异文档，提取结构化数据。
- 输出结果为 `api_difference_info.json`，保存在当前目录下。

**使用方式**:
```bash
python docs/guides/model_convert/convert_from_pytorch/tools/get_api_difference_info.py
```

---

### 2. `generate_pytorch_api_mapping.py`
**功能**:
- 整合 `api_difference_info.json`、`global_var.py`、`api_mapping.json`、`attribute_mapping.json` 生成最终映射文档 `pytorch_api_mapping_cn.md`。 其中`global_var.py`、`api_mapping.json`、`attribute_mapping.json`来自于 `PaConvert` 工具。
- **分类逻辑**:
  - 第一类 API（完全一致）: 来自 `global_var.py` 中的 `NO_NEED_CONVERT_LIST`。
  - 第二类 API（调用方式不一致）: 来自 `api_mapping.json` 与 `attribute_mapping.json`。
  - 其余类别: 从差异文档中提取。
- **参数说明**:
  - `--check`: 将映射文档保存为 `tmp_check.md`，避免覆盖原文件。

**使用方式**:
```bash
# 生成正式文档
python docs/guides/model_convert/convert_from_pytorch/tools/generate_pytorch_api_mapping.py

# 本地测试预览
python docs/guides/model_convert/convert_from_pytorch/tools/generate_pytorch_api_mapping.py --check
```

---

### 3. `validate_pytorch_api_mapping.py`
**功能**:
- 校验映射文档格式与内容，包括以下规则：
  1. 校验开头的“API 映射分类”表格与后续标题一致性。
  2. 确保每个 API 在表格中仅出现一次。
  3. 校验表格中的 API 与差异文档一一对应。
  4. 验证超链接的有效性（可选跳过）。

**使用方式**:
```bash
# 默认校验映射文档
python /workspace/paddleDocs/docs/guides/model_convert/convert_from_pytorch/tools/validate_pytorch_api_mapping.py

# 跳过 URL 校验
python /workspace/paddleDocs/docs/guides/model_convert/convert_from_pytorch/tools/validate_pytorch_api_mapping.py --skip-url-check

# 自定义映射文档路径
python /workspace/paddleDocs/docs/guides/model_convert/convert_from_pytorch/tools/validate_pytorch_api_mapping.py --file /docs/guides/model_convert/convert_from_pytorch/tools/tmp_check.md
```

---

### 4. `validate_api_difference_format.py`
**功能**:
- 根据 `pytorch_api_mapping_format_cn.md` 的格式规范，校验所有差异文档。
- 错误详情保存到 `validate_api_difference_format_error.txt`。

**使用方式**:
```bash
python docs/guides/model_convert/convert_from_pytorch/tools/validate_api_difference_format.py
```

---

### 5. `validate_api_difference_consistency.py`
**功能**:
- 校验差异文档与 `PaConvert/api_mapping.json`、`attribute_mapping.json` 的一致性，包括：
  - `Matcher` 中的 `torch api` 是否有对应映射文档。
  - `torch api` 对应的 `paddle api` 在文档与 PaConvert 中是否一致。
  - 参数映射表是否包含 `kwargs_change`。
  - 函数签名是否与 `args_list` 一致。
- 错误信息保存到 `validate_api_difference_consistency_error.txt`。

**使用方式**:
```bash
python docs/guides/model_convert/convert_from_pytorch/tools/validate_api_difference_consistency.py
```

---

### 6. `utils.py`
**功能**:
- 提供通用工具函数，包括：
  - URL 生成与校验
  - JSON 文件加载
  - 第一类 API 提取
  - 字符串处理等
---

通过本工具链，开发者可高效维护 PyTorch 与 PaddlePaddle 的 API 映射文档，确保迁移工作的准确性与一致性。
