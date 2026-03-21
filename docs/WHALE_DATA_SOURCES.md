# Whale Sound Data Sources (2025)

## Best Current Sources

### 1. **Mendeley Data Repository** ⭐ RECOMMENDED
- **Dataset**: "The active space of sperm whale codas" (2024)
- **URL**: https://data.mendeley.com/datasets/msvscjgtfp/2
- **License**: CC BY 4.0 (free for research)
- **Content**: Multi-track acoustic recordings from 80m vertical hydrophone array, tag data from 23 whale deployments
- **Format**: Research dataset with metadata
- **Access**: Direct download after free account creation

### 2. **NOAA Passive Acoustic Monitoring (PAM)** ⭐ RECOMMENDED
- **Interactive Map**: https://www.ncei.noaa.gov/maps/passive-acoustic-data/
- **Google Cloud Storage**: https://console.cloud.google.com/marketplace/details/noaa-public/passive_acoustic_monitoring
- **Contact**: pad.info@noaa.gov
- **Content**: Raw audio files (.wav, .flac) from stationary and mobile platforms
- **Coverage**: Multiple species, multiple locations, 2018-2021+ data
- **Access**: Free for research, bulk downloads available

### 3. **Macaulay Library (Cornell Lab of Ornithology)**
- **Search Portal**: https://search.macaulaylibrary.org/catalog
- **Contact**: macaulaylibrary@cornell.edu
- **Content**: World's largest wildlife sound archive, includes marine mammals
- **Access**: Free for research/education, request access (no public API)
- **Note**: Search for "sperm whale" or "Physeter macrocephalus"

### 4. **British Library Sounds**
- **Base URL**: https://sounds.bl.uk/
- **Sperm Whale Recordings**:
  - Clicks: https://sounds.bl.uk/Environment/Underwater-sounds/022M-WS0000119-0001-0001
  - Codas: https://sounds.bl.uk/Environment/Underwater-sounds/022M-WS0000120-0001-0001
  - Social sounds: https://sounds.bl.uk/Environment/Underwater-sounds/022M-WS0000121-0001-0001
- **Access**: Free streaming, may require download requests for WAV files

### 5. **Woods Hole Oceanographic Institution (WHOI)**
- **Direct WAV Downloads** (still working):
  - Click: https://www.whoi.edu/wp-content/uploads/2019/05/spermWhaleClick.wav
  - Coda: https://www.whoi.edu/wp-content/uploads/2019/05/spermWhaleCoda.wav
- **Access**: Direct download, no account needed

### 6. **DOSITS (Discovery of Sound in the Sea)** ⭐ RECOMMENDED
- **Base URL**: https://dosits.org/
- **Audio Gallery**: https://dosits.org/galleries/audio-gallery/marine-mammals/baleen-whales/humpback-whale/
- **Institution**: University of Rhode Island, Inner Space Center
- **Content**: Scientific recordings with species, location, and recording context metadata
- **Example**: Humpback whale song (Cordell Bank Canyon, CA) – direct MP3 download
- **License**: Creative Commons (Non-commercial attribution)
- **Access**: Free streaming and direct download links

### 7. **Nature Scientific Reports Datasets**
- **2025 Publication**: "Automatic detection and annotation of eastern Caribbean sperm whale codas"
- **Note**: Check paper supplementary materials for dataset links
- **URL**: https://www.nature.com/articles/s41598-025-97009-z

## How to Use These Sources

### For Quick Testing (WHOI)
```bash
# Direct download of sample files
curl -O https://www.whoi.edu/wp-content/uploads/2019/05/spermWhaleClick.wav
curl -O https://www.whoi.edu/wp-content/uploads/2019/05/spermWhaleCoda.wav
```

### For Humpback Whale (DOSITS)
```bash
# Humpback whale song (Cordell Bank Canyon, CA)
curl -O https://dosits.org/wp-content/uploads/2023/01/hump1.mp3
```

### For Research Datasets (Mendeley)
1. Create free Mendeley account
2. Visit: https://data.mendeley.com/datasets/msvscjgtfp/2
3. Click "Download" button
4. Extract and process WAV files

### For Large Collections (NOAA)
1. Use interactive map to find recordings: https://www.ncei.noaa.gov/maps/passive-acoustic-data/
2. Request bulk access via pad.info@noaa.gov
3. Or access via Google Cloud Platform (requires GCP account)

### For Academic Research (Macaulay Library)
1. Email macaulaylibrary@cornell.edu with research proposal
2. Request access to sperm whale recordings
3. They'll provide download links or API access

## Broken/Outdated Sources (from your old sources.py)
These URLs are no longer working:
- Most NOAA Fisheries sound archive pages (404)
- Scripps Institution multimedia pages (404)
- Duke Marine Lab (domain doesn't resolve)
- Cornell BRP sound pages (404)
- Many others...

## Recommendations

**For Testing/Development:**
- Use WHOI direct downloads (fastest, simplest)
- Use British Library Sounds (reliable streaming)

**For Research/Production:**
- Mendeley Data repository (structured, documented)
- NOAA PAM database (comprehensive, well-maintained)

**For Large-Scale Analysis:**
- Contact NOAA for bulk access
- Use Google Cloud Platform for NOAA data

## Next Steps

Consider creating a new downloader script that:
1. Downloads from Mendeley Data API
2. Accesses NOAA PAM via their API/cloud storage
3. Falls back to WHOI direct downloads for quick testing
4. Handles authentication and rate limiting properly
