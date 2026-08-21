#!/bin/bash
# bsrc-h3-test: attacker-controlled artifact content
pr_id="$1"
echo "## bsrc-h3-artifact-hijack-verified"
echo ""
echo "This comment was injected via attacker-controlled artifact."
echo "PR: #${pr_id} | Test: BSRC security verification"
echo ""
echo "If you see this as a github-actions[bot] comment, H3 is confirmed."
