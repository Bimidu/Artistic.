# HuggingFace Token Setup

## Quick Setup: Token in .env File

You can now store your HuggingFace token directly in the `.env` file for automatic authentication!

### Method 1: Add Token to .env Manually

1. **Get your token**:
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "artistic-asd-project")
   - Select "Write" permission
   - Copy the token

2. **Add to .env**:
   ```bash
   # Open .env file
   nano .env  # or use your preferred editor
   ```

3. **Add this line**:
   ```env
   HF_TOKEN=hf_your_token_here
   ```

4. **Save and you're done!** The app will automatically use this token.

### Method 2: Use Login Script (Auto-saves to .env)

1. **Run the login script**:
   ```bash
   python3 scripts/hf_login.py
   ```

2. **Paste your token** when prompted

3. **Say 'y'** when asked to save to .env

4. **Done!** Token is saved to both cache and .env

### How It Works

The system checks for token in this order:

1. **`.env` file** (`HF_TOKEN=...`) ← **Highest priority**
2. **Environment variable** (`HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`)
3. **Cached token** (`~/.cache/huggingface/token`)

### Security Notes

✅ **`.env` is already in `.gitignore`** - Your token won't be committed to git

⚠️ **Best Practices**:
- Never commit `.env` to git (already protected)
- Don't share your token publicly
- Use different tokens for different projects
- Rotate tokens periodically

### Verify Token is Working

```bash
# Check status
python3 scripts/cloud_sync.py status

# Should show:
# Authenticated: True
# Token source: .env file
```

### Troubleshooting

**Token not working?**
1. Check `.env` file has `HF_TOKEN=...` (no quotes needed)
2. Make sure token starts with `hf_`
3. Verify token has "Write" permission
4. Restart your app/script after adding token

**Want to use cached token instead?**
- Remove `HF_TOKEN=` line from `.env`
- Or comment it out: `# HF_TOKEN=...`
- System will use cached token from `~/.cache/huggingface/token`

### Example .env File

```env
# Cloud Storage Configuration
USE_CLOUD_STORAGE=True
CLOUD_FALLBACK_LOCAL=True

# HuggingFace Repositories
HF_DATASET_REPO=your-username/artistic-asd-datasets
HF_MODEL_REPO=your-username/artistic-asd-models

# HuggingFace Token (get from https://huggingface.co/settings/tokens)
HF_TOKEN=hf_your_actual_token_here

# Other settings...
```

That's it! Your token is now in `.env` and will be used automatically. 🎉
