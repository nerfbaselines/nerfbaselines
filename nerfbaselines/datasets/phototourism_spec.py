from nerfbaselines import register


register({
    "id": "phototourism",
    "download_dataset_function": ".phototourism:download_phototourism_dataset",
    "evaluation_protocol": {
        "evaluation_protocol_class": ".phototourism:NerfWEvaluationProtocol",
        "id": "nerfw",
    },
    "metadata": {
        "id": "phototourism",
        "name": "Photo Tourism",
        "default_metric": "psnr",
        "link": "https://phototour.cs.washington.edu/",
        "paper_link": "https://phototour.cs.washington.edu/Photo_Tourism.pdf",
        "paper_title": "Photo Tourism: Exploring Photo Collections in 3D",
        "paper_authors": ["Noah Snavely", "Steven M. Seitz", "Richard Szeliski"],
        "description": "Photo Tourism is a dataset of images of famous landmarks, such as the Sacre Coeur, the Trevi Fountain, and the Brandenburg Gate. The images were captured by tourist at different times of the day and year, images have varying lighting conditions and occlusions. The evaluation protocol is based on NeRF-W, where the image appearance embeddings are optimized on the left side of the image and the metrics are computed on the right side of the image.",
        "metrics": ["psnr", "ssim", "lpips"],
        "scenes": [{
            "id": "sacre-coeur",
            "name": "Sacre Coeur"
        }, {
            "id": "trevi-fountain",
            "name": "Trevi Fountain"
        }, {
            "id": "brandenburg-gate",
            "name": "Brandenburg Gate"
        }]
    }
})
