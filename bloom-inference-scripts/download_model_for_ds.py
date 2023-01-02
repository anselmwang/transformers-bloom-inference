from huggingface_hub import snapshot_download
snapshot_download(
    'microsoft/bloom-deepspeed-inference-fp16',
    local_files_only=False,
    cache_dir=None,
    # ignore_patterns=["*.safetensors"],
    ignore_patterns=["*.safetensors", "flax_model*", "tf_model*"],
    resume_download=True
)
