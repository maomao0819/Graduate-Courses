#!/bin/bash

# Delete all files larger than 100MB
find . -type f -size +100M -exec rm -f {} \;
find . -type f -size +100M ! -path "./.git/*" -exec rm -f {} \;
# Remove .git directories NOT in the root (depth > 1)
find . -type d -name ".git" ! -path "./.git" -exec rm -rf {} \;

echo "Cleanup completed: Removed files > 100MB and non-root .git folders."

# find . -type f -size +100M ! -path "./.git/*" -exec echo "Deleting: {}" \;
# find . -type d -name ".git" ! -path "./.git" -exec echo "Removing .git: {}" \;