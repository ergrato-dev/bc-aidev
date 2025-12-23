#!/bin/bash
# ============================================
# 🔧 Setup Script for Auto-commit on Fedora 43
# ============================================
# Run this script once to configure auto-commit
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

echo "🤖 AI Bootcamp Auto-commit Setup"
echo "================================="
echo ""

# Create logs directory
echo "📁 Creating logs directory..."
mkdir -p "$SCRIPT_DIR/logs"

# Make scripts executable
echo "🔐 Setting permissions..."
chmod +x "$SCRIPT_DIR/autocommit.sh"

# Ask for setup method
echo ""
echo "Choose setup method:"
echo "  1) Crontab (traditional)"
echo "  2) Systemd timer (modern, recommended)"
echo "  3) Both"
echo "  4) Skip (manual setup)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "📝 Setting up Crontab..."
        # Check if entry already exists
        if crontab -l 2>/dev/null | grep -q "autocommit.sh"; then
            echo "⚠️  Crontab entry already exists"
        else
            (crontab -l 2>/dev/null; echo "*/5 * * * * $SCRIPT_DIR/autocommit.sh >> $SCRIPT_DIR/logs/cron.log 2>&1") | crontab -
            echo "✅ Crontab configured!"
        fi
        ;;
    2|3)
        echo ""
        echo "📝 Setting up Systemd timer..."
        
        # Create user systemd directory
        mkdir -p "$SYSTEMD_USER_DIR"
        
        # Copy service and timer files
        cp "$SCRIPT_DIR/systemd/autocommit.service" "$SYSTEMD_USER_DIR/"
        cp "$SCRIPT_DIR/systemd/autocommit.timer" "$SYSTEMD_USER_DIR/"
        
        # Reload systemd
        systemctl --user daemon-reload
        
        # Enable and start timer
        systemctl --user enable autocommit.timer
        systemctl --user start autocommit.timer
        
        echo "✅ Systemd timer configured and started!"
        
        if [ "$choice" = "3" ]; then
            echo ""
            echo "📝 Also setting up Crontab..."
            if crontab -l 2>/dev/null | grep -q "autocommit.sh"; then
                echo "⚠️  Crontab entry already exists"
            else
                (crontab -l 2>/dev/null; echo "*/5 * * * * $SCRIPT_DIR/autocommit.sh >> $SCRIPT_DIR/logs/cron.log 2>&1") | crontab -
                echo "✅ Crontab configured!"
            fi
        fi
        ;;
    4)
        echo "⏭️  Skipping automatic setup"
        echo ""
        echo "To set up manually:"
        echo "  Crontab: crontab -e"
        echo "  Add: */5 * * * * $SCRIPT_DIR/autocommit.sh"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "================================="
echo "✅ Setup complete!"
echo ""
echo "📋 Useful commands:"
echo "  • Test script:     bash $SCRIPT_DIR/autocommit.sh"
echo "  • View cron jobs:  crontab -l"
echo "  • Timer status:    systemctl --user status autocommit.timer"
echo "  • View logs:       tail -f $SCRIPT_DIR/logs/autocommit.log"
echo ""
echo "🔧 To disable:"
echo "  • Crontab:         crontab -e (remove line)"
echo "  • Systemd:         systemctl --user stop autocommit.timer"
echo "                     systemctl --user disable autocommit.timer"
echo ""
