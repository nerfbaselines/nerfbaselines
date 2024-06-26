from ..registry import register


register(name="llff", spec={
    "load_dataset_function": ".llff:load_llff_dataset",
    "priority": 130,
    "download_dataset_function": ".llff:download_llff_dataset",
    "evaluation_protocol": "nerf",
    "metadata": {
        "id": "llff",
        "name": "LLFF",
        "metrics": ["psnr", "ssim", "lpips_vgg"],
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
