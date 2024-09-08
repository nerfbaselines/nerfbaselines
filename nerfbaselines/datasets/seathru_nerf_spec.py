from nerfbaselines import register


register({
    "id": "seathru-nerf",
    "download_dataset_function": ".seathru_nerf:download_seathru_nerf_dataset",
    "metadata": {
        "id": "seathru-nerf",
        "name": "SeaThru-NeRF",
        "description": "SeaThru-NeRF dataset contains four underwater forward-facing scenes.",
        "paper_title": "SeaThru-NeRF: Neural Radiance Fields in Scattering Media",
        "paper_authors": ["Deborah Levy", "Amit Peleg", "Naama Pearl", "Dan Rosenbaum", "Derya Akkaynak", "Tali Treibitz", "Simon Korman"],
        "paper_link": "https://openaccess.thecvf.com/content/CVPR2023/papers/Levy_SeaThru-NeRF_Neural_Radiance_Fields_in_Scattering_Media_CVPR_2023_paper.pdf",
        "link": "https://sea-thru-nerf.github.io/",
        "licenses": [{"name": "Apache 2.0","url": "https://raw.githubusercontent.com/deborahLevy130/seathru_NeRF/master/LICENSE"}],
        "metrics": ["psnr", "ssim", "lpips"],
        "default_metric": "psnr",
        "scenes": [
            {
                "id": "curasao",
                "name": "Curasao"
            },
            {
                "id": "panama",
                "name": "Panama"
            },
            {
                "id": "iui3",
                "name": "IUI3"
            },
            {
                "id": "japanese-gradens",
                "name": "Japanese Gradens"
            },
        ]
    }
})
