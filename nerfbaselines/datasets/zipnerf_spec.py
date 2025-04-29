from nerfbaselines import register


register({
    "id": "zipnerf",
    "download_dataset_function": ".zipnerf:download_zipnerf_dataset",
    "evaluation_protocol": "nerf",
    "metadata": {
        "id": "zipnerf",
        "name": "Zip-NeRF",
        "description": "ZipNeRF is a dataset with four large scenes: Berlin, Alameda, London, and NYC, (1000-2000 photos each) captured using fisheye cameras. This implementation uses undistorted images which are provided with the dataset and the downsampled resolutions are between 1392 × 793 and 2000 × 1140 depending on scene. It is recommended to use exposure modeling with this dataset if available.",
        "paper_title": "Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields",
        "paper_authors": ["Jonathan T. Barron", "Ben Mildenhall", "Dor Verbin", "Pratul P. Srinivasan", "Peter Hedman"],
        "paper_link": "https://arxiv.org/pdf/2304.06706.pdf",
        "link": "https://jonbarron.info/zipnerf/",
        "metrics": ["psnr", "ssim", "lpips_vgg"],
        "default_metric": "psnr",
        "scenes": [
            {
                "id": "alameda",
                "name": "Alameda"
            },
            {
                "id": "berlin",
                "name": "Berlin"
            },
            {
                "id": "london",
                "name": "London"
            },
            {
                "id": "nyc",
                "name": "NYC"
            }
        ]
    }
})
