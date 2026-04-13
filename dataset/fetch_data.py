"""
    This script is an example to fetch the OmniSmall dataset
        from Hugging Face. Please install associate packages
        to before executing this script
    
    For fetching the JRDB pano-track dataset, please refer 
        to instructions from the official at:
        https://jrdb.erc.monash.edu/dataset/panotrack
"""
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="xinsxins/OmniSmall",
        repo_type="dataset",
        local_dir="OmniSmall",
    )

except ImportError:
    import os
    os.system("hf download xinsxins/OmniSmall --repo-type dataset --local-dir OmniSmall")