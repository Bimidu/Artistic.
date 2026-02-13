#!/usr/bin/env python3
"""
HuggingFace Login Script

Alternative to huggingface-cli login when CLI is not available.

Usage:
    python3 scripts/hf_login.py

Author: Bimidu Gunathilake
Date: 2026-02-13
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from huggingface_hub import login, HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as e:
    print(f"❌ Error: Could not import huggingface_hub: {e}")
    print("\nPlease install it:")
    print("  pip install --upgrade huggingface-hub")
    sys.exit(1)


def main():
    """Login to HuggingFace Hub."""
    print("\n" + "="*70)
    print("  HuggingFace Hub Login")
    print("="*70 + "\n")
    
    print("To get your token:")
    print("  1. Go to: https://huggingface.co/settings/tokens")
    print("  2. Click 'New token'")
    print("  3. Name it (e.g., 'artistic-asd-project')")
    print("  4. Select 'Write' permission")
    print("  5. Copy the token")
    print()
    
    # Get token from user
    token = input("Paste your HuggingFace token here: ").strip()
    
    if not token:
        print("\n❌ No token provided. Exiting.")
        sys.exit(1)
    
    try:
        # Login with token
        print("\nAuthenticating...")
        login(token=token)
        
        # Verify authentication
        api = HfApi()
        user_info = api.whoami(token=token)
        
        print("\n" + "="*70)
        print("✅ Successfully authenticated!")
        print("="*70)
        print(f"\nLogged in as: {user_info.get('name', 'Unknown')}")
        print(f"Email: {user_info.get('email', 'Not provided')}")
        print(f"\nToken saved to: ~/.cache/huggingface/token")
        
        # Ask if user wants to save to .env
        print("\n" + "-"*70)
        save_to_env = input("Save token to .env file? (y/n): ").strip().lower()
        
        if save_to_env == 'y':
            env_file = project_root / ".env"
            if env_file.exists():
                # Read current .env
                with open(env_file, 'r') as f:
                    content = f.read()
                
                # Check if HF_TOKEN already exists
                if 'HF_TOKEN=' in content:
                    # Update existing token
                    import re
                    content = re.sub(r'HF_TOKEN=.*', f'HF_TOKEN={token}', content)
                else:
                    # Add token after HF_MODEL_REPO
                    if 'HF_MODEL_REPO=' in content:
                        content = content.replace(
                            'HF_MODEL_REPO=',
                            f'HF_MODEL_REPO=\nHF_TOKEN={token}'
                        )
                    else:
                        # Add at end
                        content += f'\nHF_TOKEN={token}\n'
                
                # Write back
                with open(env_file, 'w') as f:
                    f.write(content)
                
                print(f"✅ Token saved to .env file: {env_file}")
                print("   Note: Make sure .env is in .gitignore to keep your token secure!")
            else:
                print("⚠️  .env file not found. Creating it...")
                with open(env_file, 'w') as f:
                    f.write(f"HF_TOKEN={token}\n")
                print(f"✅ Created .env file with token: {env_file}")
        
        print("\nYou can now use cloud storage features!")
        print()
        
        return 0
        
    except HfHubHTTPError as e:
        print(f"\n❌ Authentication failed: {e}")
        if e.response.status_code == 401:
            print("   Invalid token. Please check your token and try again.")
        elif e.response.status_code == 403:
            print("   Token doesn't have required permissions.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(130)
