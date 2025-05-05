from nerfbaselines import register


register({
    "id": "hierarchical-3dgs",
    "download_dataset_function": ".hierarchical_3dgs:download_hierarchical_3dgs_dataset",
    "evaluation_protocol": "nerf",
    "metadata": {
        "id": "hierarchical-3dgs",
        "name": "Hierarchical 3DGS",
        "description": "Hierarchical 3DGS is a dataset released with H3DGS paper. We implement the two public single-chunks scenes (SmallCity, Campus) used for evaluation. To collect the dataset, authors used a bicycle helmet on which they mounted 6 GoPro HERO6 Black cameras (5 for the Campus scene). They collected SmallCity and BigCity captures on a bicycle, riding at around 6–7km/h, while Campus was captured on foot wearing the helmet. Poses were estimated using COLMAP with custom parameters and hierarchical mapper. Additinal per-chunk bundle adjustment was performed. It is recommended to use exposure modeling with this dataset",
        "paper_title": "A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets",
        "paper_authors": ["Bernhard Kerbl", "Andreas Meuleman", "Georgios Kopanas", "Michael Wimmer", "Alexandre Lanvin", "George Drettakis"],
        "paper_link": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/hierarchical-3d-gaussians_low.pdf",
        "link": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/",
        "metrics": ["psnr", "ssim", "lpips_vgg"],
        "default_metric": "psnr",
        "scenes": [
            {
                "id": "smallcity",
                "name": "SmallCity"
            },
            {
                "id": "campus",
                "name": "Campus"
            },
        ]
    }
})
