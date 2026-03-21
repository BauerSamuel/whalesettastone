"""
Simple script to download whale sounds from verified working sources.

Usage (from repository root):
    python scripts/download_whale_sounds.py [--source whoi|mendeley|british-library] [--output-dir DIR]
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import logging
import requests
from typing import Optional
import time

from src.config import WHALE_CODAS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, max_retries: int = 3) -> bool:
    """Download a file from URL with retries."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {url} (attempt {attempt}/{max_retries})...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            logger.info(f"✓ Downloaded: {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"✗ Failed to download {url} after {max_retries} attempts")
                return False
    
    return False


def download_whoi_files(output_dir: Path) -> int:
    """Download WHOI direct WAV files."""
    whoi_files = [
        {
            "url": "https://www.whoi.edu/wp-content/uploads/2019/05/spermWhaleClick.wav",
            "filename": "whoi_sperm_whale_click.wav"
        },
        {
            "url": "https://www.whoi.edu/wp-content/uploads/2019/05/spermWhaleCoda.wav",
            "filename": "whoi_sperm_whale_coda.wav"
        }
    ]
    
    downloaded = 0
    for file_info in whoi_files:
        output_path = output_dir / file_info["filename"]
        if output_path.exists():
            logger.info(f"File already exists: {output_path}, skipping...")
            continue
            
        if download_file(file_info["url"], output_path):
            downloaded += 1
    
    return downloaded


def download_mendeley_info():
    """Print information about Mendeley dataset (requires manual download)."""
    logger.info("=" * 60)
    logger.info("Mendeley Data Repository")
    logger.info("=" * 60)
    logger.info("URL: https://data.mendeley.com/datasets/msvscjgtfp/2")
    logger.info("")
    logger.info("This dataset requires:")
    logger.info("1. Free Mendeley account (create at https://www.mendeley.com)")
    logger.info("2. Visit the URL above")
    logger.info("3. Click 'Download' button")
    logger.info("4. Extract ZIP file")
    logger.info("5. Copy WAV files to your data directory")
    logger.info("")
    logger.info("License: CC BY 4.0 (free for research)")
    logger.info("=" * 60)


def download_noaa_info():
    """Print information about NOAA PAM database."""
    logger.info("=" * 60)
    logger.info("NOAA Passive Acoustic Monitoring Database")
    logger.info("=" * 60)
    logger.info("Interactive Map: https://www.ncei.noaa.gov/maps/passive-acoustic-data/")
    logger.info("Google Cloud: https://console.cloud.google.com/marketplace/details/noaa-public/passive_acoustic_monitoring")
    logger.info("")
    logger.info("For bulk downloads, contact: pad.info@noaa.gov")
    logger.info("")
    logger.info("This is a large database - use the interactive map to find")
    logger.info("specific recordings, or request bulk access via email.")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download whale sounds from verified sources"
    )
    parser.add_argument(
        "--source",
        choices=["whoi", "mendeley", "noaa", "all"],
        default="whoi",
        help="Source to download from (default: whoi)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WHALE_CODAS_DIR,
        help=f"Output directory (default: {WHALE_CODAS_DIR})"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    downloaded = 0
    
    if args.source in ["whoi", "all"]:
        logger.info("\n📥 Downloading from Woods Hole Oceanographic Institution...")
        downloaded += download_whoi_files(args.output_dir)
    
    if args.source in ["mendeley", "all"]:
        download_mendeley_info()
    
    if args.source in ["noaa", "all"]:
        download_noaa_info()
    
    if downloaded > 0:
        logger.info(f"\n✓ Successfully downloaded {downloaded} file(s) to {args.output_dir}")
    else:
        logger.info("\nNo new files downloaded (files may already exist)")


if __name__ == "__main__":
    main()
