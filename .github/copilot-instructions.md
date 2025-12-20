# PaddlePaddle 文档库 Agent 指南

本仓库（`PaddlePaddle/docs`）包含深度学习框架 PaddlePaddle（飞桨）的官方文档。它使用 Sphinx 将 ReStructuredText（`.rst`）和 Markdown（`.md`）文件生成为 HTML 文档。

## 项目结构

本仓库主要包含以下内容：

- `ci_scripts/` 中英文文档构建脚本、配置文件和 CI 脚本；
- `docs/` 中英文安装、使用、贡献指南等文档；
- `docs/api` API 参考文档的中文部分（英文部分在 `PaddlePaddle/Paddle` 仓库随源码 docstring 一起维护）。

## 项目规范

### 文档书写规范

- 由于本仓库同时包含 ReStructuredText（`.rst`）和 Markdown（`.md`）格式的文档，请根据具体文件类型遵守相应的语法规范。
- 中英文之间（语法层面）需要使用空格分隔，比如：
  - `飞桨PaddlePaddle` 应写为 `飞桨 PaddlePaddle`。 <!-- dochooks: skip-line -->
  - `[飞桨](https://www.paddlepaddle.org.cn)PaddlePaddle` 应写为 `[飞桨](https://www.paddlepaddle.org.cn) PaddlePaddle`。
- 中文文档中尽量避免使用英文标点符号，如逗号、句号、引号等，均应使用中文标点符号，英文文档中则相反，比如：
  - 英文文档中 `PaddlePaddle，is a deep learning framework，it supports multiple operating systems (Linux、macOS、Windows、etc.)。` 应写为 `PaddlePaddle, is a deep learning framework, it supports multiple operating systems (Linux, macOS, Windows, etc.).`。
  - 中文文档中 `PaddlePaddle, 是一个深度学习框架, 它支持多种操作系统 (Linux, macOS, Windows 等).` 应写为 `PaddlePaddle，是一个深度学习框架，它支持多种操作系统（Linux、macOS、Windows 等）。`。
- 在 Markdown 中应当适当为变量名、代码片段等添加反引号（`` ` ``）标记以示区分，比如：
  - `使用 paddle.to_tensor 创建张量` 应写为 `` 使用 `paddle.to_tensor` 创建张量 ``。
  - `变量 x 的值为 10` 应写为 `` 变量 `x` 的值为 10 ``。
- 所有链接应该严格有效，避免出现死链，比如：
  - 对于 reStructuredText 文件，请尽可能使用 label。如 `` :ref:`paddle.abs <cn_api_paddle_abs>` ``。
  - 对于 Markdown 文件，请使用相对路径或绝对路径链接到有效页面，如 `[安装指南](./getting_started/install_cn.md)`。

### API 文档规范

API 文档应当遵守 [API 文档书写规范](../docs/dev_guides/api_contributing_guides/api_docs_guidelines_cn.md)中的要求，以下几点应当严格遵守：

- 中文 API 文档应当与 docstring 中的英文内容严格保持语义一致（特别是 API 签名），允许适当调整表达以符合中文习惯，但不得遗漏任何参数、返回值或异常等关键信息，不得遗漏关键语句。
- 中文 API 文档中的示例代码应该使用 `COPY-FROM` 标记从英文文档中复制，确保示例代码一致，非必要不允许单独编写。
- 应当注意中文 API 文档使用语法为 ReStructuredText（`.rst`），而非 Markdown（`.md`）。请遵守 ReStructuredText 的语法规范。

## 项目检查

每次修改文档后，建议运行以下检查脚本以确保文档质量：

```bash
# 请提前通过 `uv tool install prek` 等方式安装 prek 工具
prek run --all-files
```

## Review 指南

在审查代码或文档更改时，请同时遵守以下内容：

### PR 标题检查

- PR 标题应尽可能使用 `[<type>] <description>` 的格式，其中 `<type>` 包括但不限于 `Docs`、`CI`、`API`、`CodeStyle` 等，便于快速识别 PR 类型。
- PR 标题应简洁明了地描述所做更改，避免使用模糊或通用的标题，如 `Update docs` 或 `Fix typos`，可以考虑改为 `` [Typos] Fix typos (`liunx` -> `linux`) in installation guide ``。 <!-- typos: disable-line -->
- 如果 PR 涉及多个方面的更改，建议拆分为多个 PR，每个 PR 专注于一个主题，以便于审查和合并。
- 具体 PR 具体分析，请为每个 PR 提供参考的 PR 标题，标题尽可能保持纯英文。

### PR 描述检查

- 如果涉及 API 文档变动，PR 描述中应当包含英文 docstring 的变动链接（`PaddlePaddle/Paddle` 仓库的 PR 链接），便于审查时对照检查，如未包含，请在评论中请求作者补充说明。
- 如果 PR 中包含描述中未提及的更改，审查时可根据上下文进行补充说明，确保其他审查者了解所有更改内容，如果上下文不明确，请在评论中请求作者补充说明。

### 回复方式

- 审查时请尽量使用中文回复，确保所有审查者都能理解评论内容，但在涉及具体代码片段或技术术语时，可以适当使用英文以确保准确表达。
- 保持评论简洁明了，避免冗长的解释，确保重点突出，便于作者快速理解和修改，并且尽可能给出具体的修改建议或示例代码。

### 风格检查

- 请重点检查文档书写规范和 API 文档规范部分提到的内容，确保所有更改均符合规范要求。
- 建议运行 `prek run --all-files` 来检查文档质量，确保没有遗漏任何问题，该命令同样会在 [codestyle-check](./workflows/codestyle-check.yml) 流水线中运行，如果流水线失败，请提示作者根据日志修复相关问题。
