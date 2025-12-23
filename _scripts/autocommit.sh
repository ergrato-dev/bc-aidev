#!/bin/bash
# ============================================
# 🤖 Auto-commit Script for AI Bootcamp
# ============================================
# Conventional Commits + What/For/Impact
# Runs every 5 minutes via cron
# ============================================

# Configuration
REPO_DIR="/home/epti/Documents/epti-dev/bc-channel/bc-aidev"
LOG_FILE="$REPO_DIR/_scripts/logs/autocommit.log"
BRANCH="main"

# Colors for terminal output (disabled in cron)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create logs directory if not exists
mkdir -p "$REPO_DIR/_scripts/logs"

# Timestamp
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Log function
log() {
    echo "[$(timestamp)] $1" >> "$LOG_FILE"
    echo -e "$1"
}

# Navigate to repo
cd "$REPO_DIR" || {
    log "❌ ERROR: Cannot access $REPO_DIR"
    exit 1
}

# Check if git repo
if [ ! -d ".git" ]; then
    log "❌ ERROR: Not a git repository"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)

# Check for changes
if git diff --quiet && git diff --staged --quiet; then
    log "ℹ️  No changes detected - skipping commit"
    exit 0
fi

# ============================================
# Analyze changes for smart commit message
# ============================================

# Get list of changed files
CHANGED_FILES=$(git status --porcelain)
ADDED_COUNT=$(echo "$CHANGED_FILES" | grep -c "^A\|^??" || echo 0)
MODIFIED_COUNT=$(echo "$CHANGED_FILES" | grep -c "^M\|^ M" || echo 0)
DELETED_COUNT=$(echo "$CHANGED_FILES" | grep -c "^D\|^ D" || echo 0)

# Detect primary change type based on paths
detect_type() {
    local files="$1"
    
    if echo "$files" | grep -q "1-teoria/"; then
        echo "docs"
    elif echo "$files" | grep -q "2-practicas/"; then
        echo "feat"
    elif echo "$files" | grep -q "3-proyecto/"; then
        echo "feat"
    elif echo "$files" | grep -q "0-assets/"; then
        echo "feat"
    elif echo "$files" | grep -q "5-glosario/"; then
        echo "docs"
    elif echo "$files" | grep -q "4-recursos/"; then
        echo "docs"
    elif echo "$files" | grep -q "_scripts/"; then
        echo "chore"
    elif echo "$files" | grep -q "_docs/"; then
        echo "docs"
    elif echo "$files" | grep -q ".github/"; then
        echo "chore"
    elif echo "$files" | grep -q "README"; then
        echo "docs"
    elif echo "$files" | grep -q "test\|spec"; then
        echo "test"
    elif echo "$files" | grep -q "fix\|bug\|error"; then
        echo "fix"
    else
        echo "chore"
    fi
}

# Detect scope (week number or general area)
detect_scope() {
    local files="$1"
    
    # Check for week-XX pattern
    local week=$(echo "$files" | grep -oP 'week-\d+' | head -1)
    if [ -n "$week" ]; then
        echo "$week"
        return
    fi
    
    # Check for other scopes
    if echo "$files" | grep -q "_scripts/"; then
        echo "scripts"
    elif echo "$files" | grep -q "_docs/"; then
        echo "docs"
    elif echo "$files" | grep -q "_assets/"; then
        echo "assets"
    elif echo "$files" | grep -q ".github/"; then
        echo "config"
    else
        echo "general"
    fi
}

# Get primary files changed (for WHAT)
get_primary_files() {
    local files="$1"
    echo "$files" | head -3 | awk '{print $2}' | xargs -I {} basename {} | tr '\n' ', ' | sed 's/,$//'
}

# Determine commit type and scope
COMMIT_TYPE=$(detect_type "$CHANGED_FILES")
COMMIT_SCOPE=$(detect_scope "$CHANGED_FILES")
PRIMARY_FILES=$(get_primary_files "$CHANGED_FILES")

# Calculate total changes
TOTAL_CHANGES=$((ADDED_COUNT + MODIFIED_COUNT + DELETED_COUNT))

# ============================================
# Build Conventional Commit Message
# ============================================

# Short description
if [ "$ADDED_COUNT" -gt 0 ] && [ "$MODIFIED_COUNT" -eq 0 ]; then
    ACTION="add"
elif [ "$MODIFIED_COUNT" -gt 0 ] && [ "$ADDED_COUNT" -eq 0 ]; then
    ACTION="update"
elif [ "$DELETED_COUNT" -gt 0 ] && [ "$ADDED_COUNT" -eq 0 ] && [ "$MODIFIED_COUNT" -eq 0 ]; then
    ACTION="remove"
else
    ACTION="update"
fi

# Build commit message
COMMIT_SUBJECT="${COMMIT_TYPE}(${COMMIT_SCOPE}): ${ACTION} ${TOTAL_CHANGES} files"

# Build extended body with WHAT/FOR/IMPACT
COMMIT_BODY=$(cat <<EOF

## What?
- Files changed: ${PRIMARY_FILES}
- Added: ${ADDED_COUNT} | Modified: ${MODIFIED_COUNT} | Deleted: ${DELETED_COUNT}

## For?
- Auto-commit to preserve work progress
- Branch: ${CURRENT_BRANCH}
- Timestamp: $(timestamp)

## Impact?
- Total files affected: ${TOTAL_CHANGES}
- Type: ${COMMIT_TYPE} changes in ${COMMIT_SCOPE}

---
🤖 Auto-committed by cron job
EOF
)

# ============================================
# Execute Git Commands
# ============================================

log "📦 Staging all changes..."
git add -A

log "💾 Committing with message: $COMMIT_SUBJECT"
git commit -m "$COMMIT_SUBJECT" -m "$COMMIT_BODY"

if [ $? -eq 0 ]; then
    log "✅ Commit successful!"
    
    # Optional: Auto-push (uncomment if needed)
    # log "🚀 Pushing to origin..."
    # git push origin "$CURRENT_BRANCH"
    
    # Log commit hash
    COMMIT_HASH=$(git rev-parse --short HEAD)
    log "📝 Commit hash: $COMMIT_HASH"
else
    log "❌ Commit failed!"
    exit 1
fi

log "----------------------------------------"
