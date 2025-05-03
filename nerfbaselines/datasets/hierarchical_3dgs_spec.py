from nerfbaselines import register


register({
    "id": "hierarchical-3dgs",
    "download_dataset_function": ".hierarchical_3dgs:download_hierarchical_3dgs_dataset",
    "evaluation_protocol": "nerf",
    "metadata": {
        "id": "hierarchical-3dgs",
        "name": "Hierarchical 3DGS",
        "description": "",
        "paper_title": "A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets",
        "paper_authors": ["Bernhard Kerbl", "Andreas Meuleman", "Georgios Kopanas", "Michael Wimmer", "Alexandre Lanvin", "George Drettakis"],
        "paper_link": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/hierarchical-3d-gaussians_low.pdf",
        "link": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/",
        "metrics": ["psnr", "ssim", "lpips_vgg"],
        "default_metric": "psnr",
        "scenes": [
            {
                "id": "smallcity",
                "name": "Small City"
            },
        ]
    }
})
