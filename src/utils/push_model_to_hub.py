#!/usr/bin/env python3
"""
Script to push a trained model to Hugging Face Hub
Based on official HF Hub documentation: https://huggingface.co/docs/huggingface_hub/guides/upload
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import os

from huggingface_hub import HfApi, login, create_repo, whoami

def setup_logging(verbose: bool = False):
    """Setup logging with appropriate level"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Also set huggingface_hub to show more info
    if verbose:
        logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
    
    return logging.getLogger(__name__)

def push_model_to_hub(
    model_path: str,
    repo_name: str,
    organization: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    verbose: bool = False
):
    """
    Push a model folder to Hugging Face Hub using current best practices
    
    Args:
        model_path: Local path to the model directory
        repo_name: Name of the repository on Hub (just the name, not full path)
        organization: Organization name (optional)
        private: Whether to make the repo private
        token: HF token (optional, will use HF_TOKEN env var if not provided)
        commit_message: Commit message for the push
        verbose: Enable verbose logging
    """
    
    logger = setup_logging(verbose)
    
    logger.info("🚀 Starting model upload process...")
    logger.info(f"📁 Model path: {model_path}")
    logger.info(f"📦 Repository name: {repo_name}")
    if organization:
        logger.info(f"🏢 Organization: {organization}")
    logger.info(f"🔒 Private repo: {private}")
    
    # Authenticate first
    logger.info("🔐 Authenticating with Hugging Face...")
    if token:
        logger.debug("Using provided token")
        login(token=token)
    else:
        # Try to use token from environment or HF CLI login
        try:
            logger.debug("Attempting to use existing authentication")
            login()
            logger.info("✅ Authentication successful")
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            logger.info("💡 Please run 'huggingface-cli login' or set HF_TOKEN environment variable")
            return False
    
    try:
        # Get current user info to construct proper repo_id
        logger.info("👤 Getting user information...")
        user_info = whoami()
        username = user_info['name']
        logger.info(f"📋 Authenticated as: {username}")
        
        # Verify model path exists
        model_path = Path(model_path)
        logger.info(f"📂 Checking if model path exists: {model_path}")
        if not model_path.exists():
            logger.error(f"❌ Model path does not exist: {model_path}")
            return False
        
        logger.info("✅ Model path exists")
        
        # Show what files will be uploaded
        logger.info("📋 Scanning files to upload...")
        all_files = list(model_path.rglob("*"))
        file_count = len([f for f in all_files if f.is_file()])
        total_size = sum(f.stat().st_size for f in all_files if f.is_file())
        
        logger.info(f"📊 Found {file_count} files to upload")
        logger.info(f"💾 Total size: {total_size / (1024**3):.2f} GB")
        
        if verbose:
            logger.debug("📝 Files to upload:")
            for file_path in sorted(all_files):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024**2)
                    logger.debug(f"  📄 {file_path.name} ({size_mb:.2f} MB)")
        
        # Create proper repository ID
        if organization:
            full_repo_id = f"{organization}/{repo_name}"
        else:
            full_repo_id = f"{username}/{repo_name}"
        
        logger.info(f"🏗️  Target repository: {full_repo_id}")
        
        # Create repository using HfApi
        api = HfApi()
        logger.info("🔨 Creating repository...")
        
        try:
            repo_url = create_repo(
                repo_id=full_repo_id,
                private=private,
                exist_ok=True,
                repo_type="model"
            )
            logger.info(f"✅ Repository ready: {repo_url}")
        except Exception as e:
            logger.error(f"❌ Failed to create repository: {e}")
            return False
        
        # Upload the entire folder using the HfApi
        logger.info("⬆️  Starting upload of all files...")
        logger.info("⏳ This may take a while for large models...")
        
        if verbose:
            logger.debug(f"Uploading folder: {model_path}")
            logger.debug(f"To repository: {full_repo_id}")
            logger.debug(f"Commit message: {commit_message or f'Upload model {repo_name}'}")
        
        # Use upload_folder from HfApi - this is the current recommended method
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=full_repo_id,
            repo_type="model",
            commit_message=commit_message or f"Upload model {repo_name}"
        )
        
        logger.info(f"✅ Successfully pushed model to https://huggingface.co/{full_repo_id}")
        logger.info("🎉 Upload completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error pushing model: {e}")
        if verbose:
            logger.exception("Full error traceback:")
        return False

def main():
    parser = argparse.ArgumentParser(description="Push model folder to Hugging Face Hub")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/final/rplan6_r64_a128-Llama-3.3-70B-Instruct",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Repository name on Hugging Face Hub (just the name, not username/repo)"
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="Organization name (optional). If not provided, will use your username"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face token (optional)"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        help="Custom commit message"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    success = push_model_to_hub(
        model_path=args.model_path,
        repo_name=args.repo_name,
        organization=args.organization,
        private=args.private,
        token=args.token,
        commit_message=args.commit_message,
        verbose=args.verbose
    )
    
    if success:
        print("🎉 Model push completed successfully!")
    else:
        print("💥 Model push failed!")
        exit(1)

if __name__ == "__main__":
    main()

# Usage examples:
# python src/utils/push_model_to_hub.py --repo_name "my-awesome-model" --verbose
# python src/utils/push_model_to_hub.py --repo_name "ds2d-model" --organization "your-org" -v
# python src/utils/push_model_to_hub.py --repo_name "private-model" --private --verbose