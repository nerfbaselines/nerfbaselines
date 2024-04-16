from ..registry import register


register(name="nerfstudio", spec={
    "load_dataset_function": ".nerfstudio:load_nerfstudio_dataset",
    "priority": 160,
    "download_dataset_function": ".nerfstudio:download_nerfstudio_dataset",
    "metadata": {
        "id": "nerfstudio",
        "name": "Nerfstudio",
        "description": "Nerfstudio Dataset includes 10 in-the-wild captures obtained using either a mobile phone or a mirror-less camera with a fisheye lens. We processed the data using either COLMAP or the Polycam app to obtain camera poses and intrinsic parameters.",
        "paper_title": "Nerfstudio: A Modular Framework for Neural Radiance Field Development",
        "paper_authors": ["Matthew Tancik", "Ethan Weber", "Evonne Ng", "Ruilong Li", "Brent Yi", "Justin Kerr", "Terrance Wang", "Alexander Kristoffersen", "Jake Austin", "Kamyar Salahi", "Abhik Ahuja", "David McAllister", "Angjoo Kanazawa"],
        "paper_link": "https://arxiv.org/pdf/2302.04264.pdf",
        "link": "https://nerf.studio",
        "metrics": ["psnr", "ssim", "lpips"],
        "default_metric": "psnr",
        "scenes": [
            {
                "id": "egypt",
                "name": "Egypt"
            },
            {
                "id": "person",
                "name": "person"
            },
            {
                "id": "kitchen",
                "name": "kitchen"
            },
            {
                "id": "plane",
                "name": "plane"
            },
            {
                "id": "dozer",
                "name": "dozer"
            },
            {
                "id": "floating-tree",
                "name": "floating tree"
            },
            {
                "id": "aspen",
                "name": "aspen"
            },
            {
                "id": "stump",
                "name": "stump"
            },
            {
                "id": "sculpture",
                "name": "sculpture"
            },
            {
                "id": "giannini-hall",
                "name": "Giannini Hall"
            }
        ]
    }
})
