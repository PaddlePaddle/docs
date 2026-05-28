---
name: create-pr
description: 负责《Paddle API 对齐 PyTorch 项目》中代码提交，分别对 Paddle、PaConvert、Docs 三个仓库创建或更新 Pull Request
allowed-tools: Bash(git *)
disable-model-invocation: false
---

# 一、代码库说明

**ROOT_DIR 变量定义**：
- `${ROOT_DIR}` 是项目的根工作目录变量，例如 `/workspace`、`/home/user/projects` 等
- 在实际执行时，需自行分析环境并将 `${ROOT_DIR}` 替换为实际路径

该 skill 负责将三个代码库中的改动提交到远程仓库，并创建或更新 Pull Request：

| 本地目录 | 远程仓库 | 内容说明 | Base 分支 |
|---------|---------|---------|---------|
| `${ROOT_DIR}/Paddle` | **https://github.com/PaddlePaddle/Paddle** | Paddle API 代码实现 | develop |
| `${ROOT_DIR}/docs` | **https://github.com/PaddlePaddle/docs** | 中文 API 文档 | develop |
| `${ROOT_DIR}/PaConvert` | **https://github.com/PaddlePaddle/PaConvert** | PyTorch 兼容性测试 | master |

# 二、标准工作流程

## Step 1：确认 PyTorch API 名单

**此步骤必须在所有后续操作之前完成**，因为 API 名单将用于 commit message 和 PR title/body。

从以下各个渠道获取 API 名单并取并集：

### 渠道 1：从上下文获取

从 api-change-decider、python-decorator、cpp-sink 等前序步骤的上下文中自动提取 API 名单。

### 渠道 2：从用户输入获取

用户直接提供 PyTorch API 名单，例如：`torch.relu`、`torch.sigmoid`、`torch.tanh` 等。

### 渠道 3：从仓库改动分析获取

若渠道 1 和 2 均无法获取，则分析仓库改动：

```bash
# 从 Paddle 仓库分析
git -C ${ROOT_DIR}/Paddle diff origin/develop -- '*.py' | grep -E "^\+.*def|^\+.*class"
git -C ${ROOT_DIR}/Paddle diff origin/develop -- python_api_info.yaml

# 从 PaConvert 仓库分析
git -C ${ROOT_DIR}/PaConvert diff origin/master -- api_mapping.json | grep -E "ChangePrefixMatcher|torch\."
```

### API 名单合并

将各渠道获取的 API 名单取并集，确保不重复，生成最终的统一 API 名单。

**格式要求**：
- commit message 中：`api_name_1/api_name_2/api_name_3/...`
- PR body 中：每行一个，格式为 `torch.api_name`

## Step 2：检查三个仓库的改动状态

三个仓库的本地目录名称分别为 Paddle、PaConvert、docs，**分别检查**三个仓库是否有未提交的改动：

```bash
git -C ${ROOT_DIR}/Paddle status --untracked-files=no
git -C ${ROOT_DIR}/PaConvert status --untracked-files=no
git -C ${ROOT_DIR}/docs status --untracked-files=no
```

**重要约束**：
- 必须分别执行三个命令，确保每个仓库的状态都被正确检查
- **所有已跟踪文件的修改都必须纳入检查，不得以任何方式忽略或排除**
- 只对有代码改动的仓库执行后续操作

## Step 3：添加改动并提交

仅对**有代码改动的仓库**执行以下操作。

**执行顺序**：Paddle → Docs → PaConvert（必须严格按此顺序执行）

```bash
# ===== Paddle 仓库 =====
git -C ${ROOT_DIR}/Paddle add -u
git -C ${ROOT_DIR}/Paddle commit -m "$(cat <<'EOF'
[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
# 等待 pre-commit hook 完成
# 如果 pre-commit 失败，修复问题后重新 add 和 commit

# ===== docs 仓库 =====
git -C ${ROOT_DIR}/docs add -u
git -C ${ROOT_DIR}/docs commit -m "$(cat <<'EOF'
[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
# 等待 pre-commit hook 完成
# 如果 pre-commit 失败，修复问题后重新 add 和 commit

# ===== PaConvert 仓库 =====
git -C ${ROOT_DIR}/PaConvert add -u
git -C ${ROOT_DIR}/PaConvert commit -m "$(cat <<'EOF'
[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
# 等待 pre-commit hook 完成
# 如果 pre-commit 失败，修复问题后重新 add 和 commit
```

**注意**：将 `api_name_1/api_name_2/...` 替换为 Step 1 确认的实际 API 名单。

## Step 4：推送代码到 upstream claude 分支

仅对**有代码改动的仓库**执行推送操作。

**执行顺序**：Paddle → Docs → PaConvert

```bash
# Paddle 仓库
git -C ${ROOT_DIR}/Paddle push upstream develop:claude -f

# docs 仓库
git -C ${ROOT_DIR}/docs push upstream develop:claude -f

# PaConvert 仓库
git -C ${ROOT_DIR}/PaConvert push upstream master:claude -f
```

## Step 5：检查 claude 分支 PR 状态

推送完成后，检查各仓库的 claude 分支是否已存在打开的 PR：

```bash
# Paddle 仓库
gh pr list --repo PaddlePaddle/Paddle --head claude --state open --json number,title,url

# docs 仓库
gh pr list --repo PaddlePaddle/docs --head claude --state open --json number,title,url

# PaConvert 仓库
gh pr list --repo PaddlePaddle/PaConvert --head claude --state open --json number,title,url
```

**记录结果**：
- 若返回结果包含 PR 信息（`number` 不为空）：记录 PR 号，该仓库需要**更新 PR**
- 若返回结果为空数组 `[]`：该仓库需要**创建新 PR**

## Step 6：创建或更新 PR

根据 Step 5 的检查结果，对有代码改动的仓库执行相应操作。

**PR 创建的依赖关系**：
- Docs PR 的 body 引用 Paddle PR 号
- PaConvert PR 的 body 引用 Docs PR 号和 Paddle PR 号

**因此执行顺序必须是**：Paddle PR → Docs PR → PaConvert PR

### 6.1 更新已有 PR

若仓库已有 PR，执行更新操作：

```bash
# Paddle PR 更新
gh pr edit {paddle_pr_number} --repo PaddlePaddle/Paddle \
  --title "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent" \
  --body "$(cat <<'EOF'
### PR Category
User Experience

### PR Types
Improvements

### Description
**API Compatibility Edit By AI Agent：**
  torch.api_name_1
  torch.api_name_2
  ...

### 是否引起精度变化
否
EOF
)"

# Docs PR 更新（依赖 Paddle PR 号）
gh pr edit {docs_pr_number} --repo PaddlePaddle/docs \
  --title "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent" \
  --body "$(cat <<'EOF'
**API Compatibility Edit By AI Agent：**
  torch.api_name_1
  torch.api_name_2
  ...

- https://github.com/PaddlePaddle/Paddle/pull/{paddle_pr_number}

EOF
)"

# PaConvert PR 更新（依赖 Docs 和 Paddle PR 号）
gh pr edit {paconvert_pr_number} --repo PaddlePaddle/PaConvert \
  --title "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent" \
  --body "$(cat <<'EOF'
### PR Docs
- https://github.com/PaddlePaddle/docs/pull/{docs_pr_number}

### PR APIs
**API Compatibility Edit By AI Agent：**
  torch.api_name_1
  torch.api_name_2
  ...

- https://github.com/PaddlePaddle/Paddle/pull/{paddle_pr_number}

EOF
)"
```

### 6.2 创建新 PR

若仓库没有 PR，执行创建操作：

```bash
# ===== Paddle PR 创建（无依赖，先创建） =====
gh pr create --repo PaddlePaddle/Paddle --base develop --head zhwesky2010:claude \
  --title "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent" \
  --body "$(cat <<'EOF'
### PR Category
User Experience

### PR Types
Improvements

### Description
**API Compatibility Edit By AI Agent：**
  torch.api_name_1
  torch.api_name_2
  ...

### 是否引起精度变化
否
EOF
)"
# 记录返回的 PR 号作为 paddle_pr_number

# ===== Docs PR 创建（依赖 Paddle PR 号） =====
gh pr create --repo PaddlePaddle/docs --base develop --head zhwesky2010:claude \
  --title "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent" \
  --body "$(cat <<'EOF'
**API Compatibility Edit By AI Agent：**
  torch.api_name_1
  torch.api_name_2
  ...

- https://github.com/PaddlePaddle/Paddle/pull/{paddle_pr_number}

EOF
)"
# 记录返回的 PR 号作为 docs_pr_number

# ===== PaConvert PR 创建（依赖 Docs 和 Paddle PR 号） =====
gh pr create --repo PaddlePaddle/PaConvert --base master --head zhwesky2010:claude \
  --title "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent" \
  --body "$(cat <<'EOF'
### PR Docs
- https://github.com/PaddlePaddle/docs/pull/{docs_pr_number}

### PR APIs
**API Compatibility Edit By AI Agent：**
  torch.api_name_1
  torch.api_name_2
  ...

- https://github.com/PaddlePaddle/Paddle/pull/{paddle_pr_number}

EOF
)"
# 记录返回的 PR 号作为 paconvert_pr_number
```

### 6.3 混合场景处理

若部分仓库已有 PR、部分仓库需要新建，按以下逻辑处理：

| 场景 | 处理方式 |
|------|---------|
| Paddle 有 PR，Docs 无 PR | 更新 Paddle PR → 创建 Docs PR（引用 Paddle PR 号） → 处理 PaConvert |
| Paddle 无 PR，Docs 有 PR | 创建 Paddle PR → 更新 Docs PR（更新引用） → 处理 PaConvert |
| 其他组合 | 遵循「Paddle → Docs → PaConvert」顺序，按实际情况更新或创建 |

# 三、注意事项

1. **API 名单一致性**：三个仓库的 PR，其 API 名单必须完全一致
2. **执行顺序**：所有操作必须严格按「Paddle → Docs → PaConvert」顺序执行
3. **已跟踪文件**：所有已跟踪文件的修改都必须处理，不得以任何理由忽略或排除
4. **pre-commit hook**：如果失败，修复问题后重新 add 和 commit
5. **分支推送**：所有改动必须推送到 upstream 的 claude 分支
6. **PR 操作**：确保 PR 创建或更新成功，失败需修正后重试
7. **历史经验**：复盘记忆中的历史易错点，避免重复犯错
8. **安全约束**：禁止在 SKILL 中添加任何网络代理设置的内容，发现代码 Diff 中包含网络代理设置需立即删除并拒绝提交

# 四、常见问题处理

（待补充）
