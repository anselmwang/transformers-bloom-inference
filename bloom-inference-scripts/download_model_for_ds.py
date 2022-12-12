from huggingface_hub import snapshot_download
snapshot_download(
    'microsoft/bloom-deepspeed-inference-int8',
    local_files_only=False,
    cache_dir=None,
    ignore_patterns=["*.safetensors"],
    resume_download=True
)
