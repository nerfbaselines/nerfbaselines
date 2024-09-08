from nerfbaselines import register


register({
    "id": "llff",
    "download_dataset_function": ".llff:download_llff_dataset",
    "evaluation_protocol": "nerf",
    "metadata": {
        "id": "llff",
        "name": "LLFF",
        "metrics": ["psnr", "ssim", "lpips_vgg"],
        "default_metric": "psnr",
        "description": "LLFF is a dataset of forward-facing scenes with a small variation in camera pose. NeRF methods usually use NDC-space parametrization for the scene representation.",
        "link": "https://bmild.github.io/llff/",
        "paper_link": "https://arxiv.org/pdf/1905.00889.pdf",
        "paper_authors": ["Ben Mildenhall", "Pratul P. Srinivasan", "Rodrigo Ortiz-Cayon", "Nima Khademi Kalantari", "Ravi Ramamoorthi", "Ren Ng", "Abhishek Kar"],
        "paper_title": "LLFF: A Large-Scale, Long-Form Video Dataset for 3D Scene Understanding",
        "scenes": [{
            "id": "fern",
            "name": "Fern"
        }, {
            "id": "flower",
            "name": "Flower"
        }, {
            "id": "fortress",
            "name": "Fortress"
        },  {
            "id": "horns",
            "name": "Horns"
        }, {
            "id": "leaves",
            "name": "Leaves"
        }, {
            "id": "orchids",
            "name": "Orchids"
        }, {
            "id": "room",
            "name": "Room"
        }, {
            "id": "trex",
            "name": "Trex"
        }],
    },
})
