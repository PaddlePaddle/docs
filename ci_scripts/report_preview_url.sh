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

    # Check if file ends with _en (English version)
    # _en.rst will only reach English preview, _en.md can be reached in both English and Chinese previews
    # To Simplify, we treat all _en files as English version
    if [[ "$path_no_ext" == *_en ]]; then
        local base_url="http://preview-pr-${pr_id}.paddle-docs-preview.paddlepaddle.org.cn/documentation/docs/en/"
    else
        # Use /zh/ path for Chinese version
        local base_url="http://preview-pr-${pr_id}.paddle-docs-preview.paddlepaddle.org.cn/documentation/docs/zh/"
    fi

    local final_url="${base_url}${path_no_ext}.html"
    echo "$final_url"
}

# Use merge-base to find the common ancestor between PR branch and develop
# This ensures we only get changes from this PR, excluding commits merged to develop after PR creation
BASE_COMMIT=$(git merge-base HEAD develop 2>/dev/null || echo "develop")

mapfile -t all_git_files < <(git diff --name-only --diff-filter=ACMR "$BASE_COMMIT" | sed 's#^docs/##')

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
<summary>📚 本次 PR 文档预览链接（点击展开）</summary>

<table>
<tr>
<td>
ℹ️ <b>预览提醒</b><br>
请等待 <code>Docs-NEW</code> 流水线运行完成后再点击预览链接，否则可能会看到旧版本内容或遇到链接无法访问的情况。
</td>
</tr>
</table>

$(printf '%s\n' "${output_lines[@]}")

</details>
EOF
fi
