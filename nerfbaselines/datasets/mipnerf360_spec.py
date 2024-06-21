from ..registry import register


register(name="mipnerf360", spec={
    "load_dataset_function": ".mipnerf360:load_mipnerf360_dataset",
    "priority": 180,
    "download_dataset_function": ".mipnerf360:download_mipnerf360_dataset",
    "evaluation_protocol": "nerf",
    "metadata": {
        "id": "mipnerf360",
        "name": "Mip-NeRF 360",
        "description": "Mip-NeRF 360 is a collection of four indoor and five outdoor object-centric scenes. The camera trajectory is an orbit around the object with fixed elevation and radius. The test set takes each n-th frame of the trajectory as test views.",
        "paper_title": "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields",
        "paper_authors": ["Jonathan T. Barron", "Ben Mildenhall", "Dor Verbin", "Pratul P. Srinivasan", "Peter Hedman"],
        "paper_link": "https://arxiv.org/pdf/2111.12077.pdf",
        "link": "https://jonbarron.info/mipnerf360/",
        "metrics": ["psnr", "ssim", "lpips_vgg"],
        "default_metric": "psnr",
        "scenes": [
            {
                "id": "garden",
                "name": "garden"
            },
            {
                "id": "bicycle",
                "name": "bicycle"
            },
            {
                "id": "flowers",
                "name": "flowers"
            },
            {
                "id": "treehill",
                "name": "treehill"
            },
            {
                "id": "stump",
                "name": "stump"
            },
            {
                "id": "kitchen",
                "name": "kitchen"
            },
            {
                "id": "bonsai",
                "name": "bonsai"
            },
            {
                "id": "counter",
                "name": "counter"
            },
            {
                "id": "room",
                "name": "room"
            }
        ]
    }
})
