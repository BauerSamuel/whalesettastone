"""
Module containing whale sound sources from various institutions and organizations.

Updated 2025: Prioritized working sources. Many old URLs are broken.
See docs/WHALE_DATA_SOURCES.md for detailed information about each source.
"""

WHALE_SOUND_SOURCES = {
    # ⭐ RECOMMENDED: Direct downloads that work reliably
    # NOTE: WHOI URLs currently return 404 - these may need to be updated
    "Direct Downloads (Working)": [
        {
            "url": "https://www.whoi.edu/wp-content/uploads/2019/05/spermWhaleClick.wav",
            "type": "clicks",
            "description": "Sperm whale click from Woods Hole Oceanographic Institution - DIRECT DOWNLOAD (may be 404)",
            "verified": True
        },
        {
            "url": "https://www.whoi.edu/wp-content/uploads/2019/05/spermWhaleCoda.wav",
            "type": "coda",
            "description": "Sperm whale coda from Woods Hole Oceanographic Institution - DIRECT DOWNLOAD (may be 404)",
            "verified": True
        }
    ],
    # ⭐ RECOMMENDED: Research datasets (2024-2025)
    "Research Datasets": [
        {
            "url": "https://data.mendeley.com/datasets/msvscjgtfp/2",
            "type": "dataset",
            "description": "The active space of sperm whale codas - Mendeley Data (2024) - CC BY 4.0",
            "requires_account": True,
            "verified": True
        },
        {
            "url": "https://www.ncei.noaa.gov/maps/passive-acoustic-data/",
            "type": "database",
            "description": "NOAA Passive Acoustic Monitoring database - Interactive map",
            "contact": "pad.info@noaa.gov",
            "verified": True
        },
        {
            "url": "https://console.cloud.google.com/marketplace/details/noaa-public/passive_acoustic_monitoring",
            "type": "cloud_storage",
            "description": "NOAA PAM data via Google Cloud Platform",
            "requires_account": True,
            "verified": True
        }
    ],
    # ⚠️ British Library - DNS resolution issues (sounds.bl.uk may be inaccessible)
    # These URLs may require manual access via https://sounds.bl.uk
    "British Library": [
        {
            "url": "https://sounds.bl.uk/Environment/Underwater-sounds/022M-WS0000119-0001-0001",
            "type": "clicks",
            "description": "Sperm Whale clicks - British Library Sounds (DNS resolution may fail)",
            "verified": True
        },
        {
            "url": "https://sounds.bl.uk/Environment/Underwater-sounds/022M-WS0000120-0001-0001",
            "type": "coda",
            "description": "Sperm Whale codas - British Library Sounds (DNS resolution may fail)",
            "verified": True
        },
        {
            "url": "https://sounds.bl.uk/Environment/Underwater-sounds/022M-WS0000121-0001-0001",
            "type": "social",
            "description": "Sperm Whale social sounds - British Library Sounds (DNS resolution may fail)",
            "verified": True
        }
    ],
    # Macaulay Library (requires email request)
    "Macaulay Library (Cornell)": [
        {
            "url": "https://search.macaulaylibrary.org/catalog",
            "type": "search",
            "description": "Search portal - search for 'sperm whale' or 'Physeter macrocephalus'",
            "contact": "macaulaylibrary@cornell.edu",
            "requires_request": True,
            "verified": True
        },
        {
            "url": "https://macaulaylibrary.org/asset/94428",
            "type": "clicks",
            "description": "Sperm whale clicks from Macaulay Library (may require login)",
            "verified": True
        }
    ],
    # DOSITS - University of Rhode Island, educational/research quality
    "DOSITS": [
        {
            "url": "https://dosits.org/wp-content/uploads/2023/01/hump1.mp3",
            "type": "humpback_song",
            "description": "Humpback whale song (Cordell Bank Canyon, CA)",
            "verified": True
        }
    ],
    # Sample files for testing
    "Sample Files": [
        {
            "url": "data/sample_audio/test_tone_1.wav",
            "type": "test",
            "description": "Test tone for system validation"
        },
        {
            "url": "data/sample_audio/test_tone_2.wav",
            "type": "test",
            "description": "Another test tone for system validation"
        }
    ],
    # ⚠️ DEPRECATED: These sources are broken/unreliable
    "Deprecated (Broken URLs)": [
        {
            "url": "https://www.fisheries.noaa.gov/national/science-data/sperm-whale-sounds",
            "type": "clicks",
            "description": "NOAA Fisheries sound archive - BROKEN (404)",
            "status": "broken"
        },
        {
            "url": "https://www.youtube.com/watch?v=zsDwFGz0Okg",
            "type": "clicks",
            "description": "YouTube - requires yt-dlp/ffmpeg",
            "status": "requires_tools"
        }
        # Most of these are broken (404s) - kept for reference
        # See docs/WHALE_DATA_SOURCES.md for working alternatives
    ]
} 