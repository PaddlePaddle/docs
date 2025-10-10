#!/bin/bash

pr_id="$1"

if [ -z "$pr_id" ]; then
    echo "Error: Pull Request ID is not provided."
    exit 1
fi

generate_preview_url() {
    local file_path="$1"
    local pr_id="$2"
    local path_no_ext="${file_path%.*}"
    local base_url="http://preview-pr-${pr_id}.paddle-docs-preview.paddlepaddle.org.cn/documentation/docs/zh/"
    local final_url="${base_url}${path_no_ext}.html"
    echo "$final_url"
}

mapfile -t all_git_files < <(git diff --name-only --diff-filter=ACMR develop | sed 's#^docs/##')

output_lines=()

for file in "${all_git_files[@]}"; do
    if [[ "$file" == *.rst || "$file" == *.md ]]; then
        url=$(generate_preview_url "$file" "$pr_id")
        output_lines+=("- \`docs/${file}\`: [点击预览](${url})")
    fi
done


if [ ${#output_lines[@]} -gt 0 ]; then
    cat <<-EOF
<details>
<summary>📚 本次 PR 文档预览链接 (点击展开)</summary>

以下是本次 PR 中变更文档的预览链接：

$(printf '%s\n' "${output_lines[@]}")

</details>
EOF
fi
