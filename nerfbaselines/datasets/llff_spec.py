from ..registry import register


register(name="llff", spec={
    "load_dataset_function": ".llff:load_llff_dataset",
    "priority": 130,
    "download_dataset_function": ".llff:download_llff_dataset",
    "evaluation_protocol": "nerf",
})
