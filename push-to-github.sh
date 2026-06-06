#!/bin/bash
# Run this script to create the GitHub repo and push your blog.
# You'll need to authenticate with GitHub first.

set -e

echo "Creating repo and pushing..."
gh repo create LLM_PostTraining --public --source=. --remote=origin --push

echo ""
echo "Done! Your blog will be at: https://AyanKumarBhunia.github.io/LLM_PostTraining/"
echo ""
echo "Next: Go to https://github.com/AyanKumarBhunia/LLM_PostTraining/settings/pages"
echo "      and set Source to 'GitHub Actions' to enable the site."
