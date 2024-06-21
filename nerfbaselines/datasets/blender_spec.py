from ..registry import register


register(name="blender", spec={
    "load_dataset_function": ".blender:load_blender_dataset",
    "priority": 150,
    "download_dataset_function": ".blender:download_blender_dataset",
    "evaluation_protocol": "nerf",
    "metadata": {
        "id": "blender",
        "name": "Blender",
        "description": "Blender (nerf-synthetic) is a synthetic dataset used to benchmark NeRF methods. It consists of 8 scenes of an object placed on a white background. Cameras are placed on a semi-sphere around the object.",
        "paper_title": "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
        "paper_authors": ["Ben Mildenhall", "Pratul P. Srinivasan", "Matthew Tancik", "Jonathan T. Barron", "Ravi Ramamoorthi", "Ren Ng"],
        "paper_link": "https://arxiv.org/pdf/2003.08934.pdf",
        "link": "https://www.matthewtancik.com/nerf",
        "metrics": ["psnr", "ssim", "lpips_vgg"],
        "default_metric": "psnr",
        "scenes": [
            {
                "id": "lego",
                "name": "lego"
            },
            {
                "id": "drums",
                "name": "drums"
            },
            {
                "id": "ficus",
                "name": "ficus"
            },
            {
                "id": "hotdog",
                "name": "hotdog"
            },
            {
                "id": "materials",
                "name": "materials"
            },
            {
                "id": "mic",
                "name": "mic"
            },
            {
                "id": "ship",
                "name": "ship"
            },
            {
                "id": "chair",
                "name": "chair"
            }
        ]
    }
})
