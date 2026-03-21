"""
Helper module to download whale sounds from various sources for use in the Streamlit app.
"""

import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import io

from .sources import WHALE_SOUND_SOURCES
from .config import WHALE_CODAS_DIR

logger = logging.getLogger(__name__)


def get_available_sources() -> Dict[str, List[Dict]]:
    """Get sources that can be automatically downloaded (have direct WAV URLs)."""
    available = {}
    
    for source_name, source_list in WHALE_SOUND_SOURCES.items():
        # Skip deprecated sources
        if "Deprecated" in source_name:
            continue
        
        downloadable_items = []
        for item in source_list:
            # Check if it's a direct download (has URL and is verified)
            if isinstance(item, dict):
                url = item.get("url", "")
                verified = item.get("verified", False)
                requires_account = item.get("requires_account", False)
                requires_request = item.get("requires_request", False)
                
                # Only include direct downloads that don't require accounts/requests
                # For local files (sample files), don't require verified flag
                is_local_file = not url.startswith("http")
                if (verified or is_local_file) and not requires_account and not requires_request:
                    # Include WAV URLs or URLs that might contain WAV files
                    # Also include any URL that ends with common audio extensions
                    # Note: Some URLs (like bl.uk, whoi.edu) may be broken but we include them
                    # so users can see the errors and we can update them later
                    if (url.endswith(".wav") or 
                        url.endswith(".mp3") or 
                        "whoi.edu" in url or 
                        "bl.uk" in url or 
                        is_local_file):
                        downloadable_items.append(item)
        
        if downloadable_items:
            available[source_name] = downloadable_items
    
    return available


def download_from_url(url: str, output_path: Optional[Path] = None, max_retries: int = 3) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Download a file from URL and return bytes, or save to file if output_path provided.
    
    Returns:
        Tuple of (content bytes or None, error message or None)
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {url} (attempt {attempt}/{max_retries})...")
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            
            # Check status code first
            if response.status_code == 404:
                error_msg = f"File not found (404): {url}"
                logger.error(error_msg)
                return None, error_msg
            
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type and 'audio' not in content_type:
                # Check if it's actually HTML content (might be a 404 page)
                content_preview = b""
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        content_preview += chunk
                        if len(content_preview) > 5000:  # Check first 5KB
                            break
                
                if b'<html' in content_preview.lower() or b'<!doctype' in content_preview.lower():
                    error_msg = f"URL returned HTML page instead of audio file (likely 404): {url}"
                    logger.error(error_msg)
                    return None, error_msg
            
            # Reset response to read full content
            content = b""
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Need to make a new request since we consumed the stream
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            content = response.content
            
            if len(content) == 0:
                error_msg = f"Downloaded empty file from {url}"
                logger.warning(error_msg)
                return None, error_msg
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(content)
                logger.info(f"✓ Saved {len(content)} bytes to: {output_path}")
            
            logger.info(f"✓ Downloaded {len(content)} bytes from {url}")
            return content, None
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error {e.response.status_code}: {url}"
            logger.warning(f"Attempt {attempt} failed: {error_msg}")
            if attempt < max_retries:
                import time
                time.sleep(2 ** attempt)
            else:
                logger.error(f"✗ Failed to download {url} after {max_retries} attempts: {error_msg}")
                return None, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {e}"
            logger.warning(f"Attempt {attempt} failed: {error_msg}")
            if attempt < max_retries:
                import time
                time.sleep(2 ** attempt)
            else:
                logger.error(f"✗ Failed to download {url} after {max_retries} attempts: {error_msg}")
                return None, error_msg
    
    return None, "Max retries exceeded"


def download_source_files(source_name: str, cache_dir: Optional[Path] = None) -> Tuple[List[io.BytesIO], List[str]]:
    """
    Download files from a source and return them as BytesIO objects (like uploaded files).
    
    Args:
        source_name: Name of the source from WHALE_SOUND_SOURCES
        cache_dir: Optional directory to cache downloaded files
    
    Returns:
        Tuple of (list of BytesIO objects, list of error messages)
    """
    errors = []
    if cache_dir is None:
        cache_dir = WHALE_CODAS_DIR / "cached_sources"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    available_sources = get_available_sources()
    
    if source_name not in available_sources:
        error_msg = f"Source '{source_name}' not available for automatic download"
        logger.error(error_msg)
        return [], [error_msg]
    
    downloaded_files = []
    source_items = available_sources[source_name]
    
    for item in source_items:
        url = item.get("url", "")
        if not url:
            errors.append(f"Item missing URL: {item.get('description', 'Unknown item')}")
            continue
        
        # Handle local files (sample files)
        if not url.startswith("http"):
            # Resolve relative paths from project root
            from .config import BASE_DIR
            local_path = Path(url)
            if not local_path.is_absolute():
                # Try relative to BASE_DIR first, then project root (BASE_DIR.parent)
                base_dir_path = BASE_DIR / local_path
                if base_dir_path.exists():
                    local_path = base_dir_path
                else:
                    # Try project root
                    project_root_path = BASE_DIR.parent / local_path
                    if project_root_path.exists():
                        local_path = project_root_path
            
            if local_path.exists():
                logger.info(f"Loading local file: {local_path}")
                try:
                    with open(local_path, 'rb') as f:
                        content = f.read()
                    file_obj = io.BytesIO(content)
                    file_obj.name = local_path.name
                    downloaded_files.append(file_obj)
                except Exception as e:
                    error_msg = f"Failed to read local file {local_path}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            else:
                error_msg = f"Local file not found: {local_path}"
                logger.warning(error_msg)
                errors.append(error_msg)
            continue
        
        # Check cache first
        filename = url.split("/")[-1] or f"file_{len(downloaded_files)}.wav"
        # Clean filename - remove query parameters
        if "?" in filename:
            filename = filename.split("?")[0]
        # Preserve original audio extension (.wav, .mp3, etc.)
        if not any(filename.lower().endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg")):
            filename += ".wav"
        
        cache_path = cache_dir / f"{source_name.replace(' ', '_')}_{filename}"
        
        # Download if not cached
        if not cache_path.exists():
            content, download_error = download_from_url(url, output_path=cache_path)
            if download_error:
                errors.append(download_error)
            if content is None:
                continue
            if len(content) == 0:
                error_msg = f"Downloaded empty file from {url}"
                errors.append(error_msg)
                cache_path.unlink(missing_ok=True)  # Remove empty cache file
                continue
        else:
            logger.info(f"Using cached file: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    content = f.read()
                if len(content) == 0:
                    # Cache file is empty, re-download
                    cache_path.unlink()
                    content, download_error = download_from_url(url, output_path=cache_path)
                    if download_error:
                        errors.append(download_error)
                    if content is None or len(content) == 0:
                        error_msg = f"Failed to download {url}"
                        errors.append(error_msg)
                        continue
            except Exception as e:
                error_msg = f"Failed to read cached file {cache_path}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
        
        # Create BytesIO object (like Streamlit uploaded files)
        try:
            file_obj = io.BytesIO(content)
            file_obj.name = filename
            downloaded_files.append(file_obj)
        except Exception as e:
            error_msg = f"Failed to create file object for {filename}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    logger.info(f"Downloaded {len(downloaded_files)} file(s) from {source_name}")
    if errors:
        logger.warning(f"Encountered {len(errors)} error(s) during download")
    return downloaded_files, errors


def get_source_display_name(source_name: str) -> str:
    """Get a user-friendly display name for a source."""
    display_names = {
        "Direct Downloads (Working)": "Woods Hole Oceanographic Institution (WHOI)",
        "British Library": "British Library Sounds",
        "DOSITS": "DOSITS (Discovery of Sound in the Sea)",
        "Sample Files": "Local Sample Files"
    }
    return display_names.get(source_name, source_name)
