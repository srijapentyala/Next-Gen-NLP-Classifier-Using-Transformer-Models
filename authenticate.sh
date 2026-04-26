#!/bin/bash

# GitHub Token Authentication Helper for macOS
# This script helps you authenticate git with your GitHub token

echo "=========================================="
echo "🔐 GitHub Token Authentication Setup"
echo "=========================================="
echo ""

echo "Step 1: Create Your Token"
echo "========================"
echo "1. Open: https://github.com/settings/tokens"
echo "2. Click 'Generate new token (classic)'"
echo "3. Fill in:"
echo "   - Token name: data-mining-project-push"
echo "   - Expiration: 90 days"
echo "   - Scope: Check ONLY 'repo'"
echo "4. Click 'Generate token'"
echo "5. COPY the token (starts with ghp_)"
echo ""

read -p "Once you've created and copied your token, press Enter to continue..."

echo ""
echo "Step 2: You'll now be prompted for your credentials"
echo "====================================================="
echo "When git asks:"
echo "  • Username: Enter your GitHub username"
echo "  • Password: Paste your token (Ctrl+V or Cmd+V)"
echo ""

read -p "Ready to push? Press Enter..."

echo ""
echo "Pushing to GitHub..."
cd /Users/srijapentyala/Downloads/DATA\ MINING\ PROJECT
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Push completed!"
    echo "=========================================="
    echo ""
    echo "Your token has been saved to Keychain."
    echo "Future pushes will work automatically."
    echo ""
    echo "Verify at:"
    echo "https://github.com/srijapentyala/Next-Gen-NLP-Classifier-Using-Transformer-Models/tree/main/checkpoints"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Push failed. Check your token and try again."
    echo "=========================================="
    echo ""
    echo "Common issues:"
    echo "1. Token was pasted incorrectly (make sure it starts with 'ghp_')"
    echo "2. Token has incorrect permissions (must have 'repo' scope)"
    echo "3. Token has expired (create a new one)"
    echo ""
fi
