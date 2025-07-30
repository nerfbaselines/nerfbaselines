from nerfbaselines import register


register({
    "id": "mipnerf360-sparse",
    "download_dataset_function": ".mipnerf360_sparse:download_mipnerf360_sparse_dataset",
    "evaluation_protocol": "nerf",
    "metadata": {
        "id": "mipnerf360-sparse",
        "name": "Mip-NeRF 360 Sparse",
        "description": "Modified Mip-NeRF 360 dataset with small train set (12 or 24) views. The dataset is used to evaluate sparse-view NVS methods.",
        "paper_title": "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields",
        "paper_authors": ["Jonathan T. Barron", "Ben Mildenhall", "Dor Verbin", "Pratul P. Srinivasan", "Peter Hedman"],
        "paper_link": "https://arxiv.org/pdf/2111.12077.pdf",
        "link": "https://jonbarron.info/mipnerf360/",
        "metrics": ["psnr", "ssim", "lpips_vgg"],
        "default_metric": "psnr",
        "scenes": [
            { "id": "garden-n12", "name": "garden n12" },
            { "id": "bicycle-n12", "name": "bicycle n12" },
            { "id": "flowers-n12", "name": "flowers n12" },
            { "id": "treehill-n12", "name": "treehill n12" },
            { "id": "stump-n12", "name": "stump n12" },
            { "id": "kitchen-n12", "name": "kitchen n12" },
            { "id": "bonsai-n12", "name": "bonsai n12" },
            { "id": "counter-n12", "name": "counter n12" },
            { "id": "room-n12", "name": "room n12" },
            { "id": "garden-n24", "name": "garden n24" },
            { "id": "bicycle-n24", "name": "bicycle n24" },
            { "id": "flowers-n24", "name": "flowers n24" },
            { "id": "treehill-n24", "name": "treehill n24" },
            { "id": "stump-n24", "name": "stump n24" },
            { "id": "kitchen-n24", "name": "kitchen n24" },
            { "id": "bonsai-n24", "name": "bonsai n24" },
            { "id": "counter-n24", "name": "counter n24" },
            { "id": "room-n24", "name": "room n24" }
        ]
    }
})
