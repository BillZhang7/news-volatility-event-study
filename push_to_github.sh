#!/bin/bash
# Push this repo to GitHub (after creating empty repo on github.com)
# Usage: bash push_to_github.sh https://github.com/BillZhang7/news-volatility-event-study.git

set -e
cd "$(dirname "$0")"

if [ -z "$1" ]; then
  echo "Usage:"
  echo "  bash push_to_github.sh https://github.com/YOUR_USERNAME/REPO_NAME.git"
  exit 1
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$1"
git push -u origin main
echo ""
echo "Done: ${1%.git}"
