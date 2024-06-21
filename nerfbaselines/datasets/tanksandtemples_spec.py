from ..registry import register


register(name="tanksandtemples", spec={
    "load_dataset_function": ".tanksandtemples:load_tanksandtemples_dataset",
    "priority": 140,
    "download_dataset_function": ".tanksandtemples:download_tanksandtemples_dataset",
    "metadata": {
        "id": "tanksandtemples",
        "name": "Tanks and Temples",
        "description": "Tanks and Temples is a benchmark for image-based 3D reconstruction. The benchmark sequences were acquired outside the lab, in realistic conditions. Ground-truth data was captured using an industrial laser scanner. The benchmark includes both outdoor scenes and indoor environments. The dataset is split into three subsets: training, intermediate, and advanced.",
        "paper_title": "Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction",
        "paper_authors": ["Arno Knapitsch", "Jaesik Park", "Qian-Yi Zhou", "Vladlen Koltun"],
        "paper_link": "https://storage.googleapis.com/t2-downloads/paper/tanks-and-temples.pdf",
        "link": "https://www.tanksandtemples.org/",
        "metrics": ["psnr", "ssim", "lpips"],
        "default_metric": "psnr",
        "scenes": [
            {"id": "auditorium", "name": "auditorium"},
            {"id": "ballroom", "name": "ballroom"},
            {"id": "courtroom", "name": "courtroom"},
            {"id": "museum", "name": "museum"},
            {"id": "palace", "name": "palace"},
            {"id": "temple", "name": "temple"},

            {"id": "family", "name": "family"},
            {"id": "francis", "name": "francis"},
            {"id": "horse", "name": "horse"},
            {"id": "lighthouse", "name": "lighthouse"},
            {"id": "m60", "name": "m60"},
            {"id": "panther", "name": "panther"},
            {"id": "playground", "name": "playground"},
            {"id": "train", "name": "train"},

            {"id": "barn", "name": "barn"},
            {"id": "caterpillar", "name": "caterpillar"},
            {"id": "church", "name": "church"},
            {"id": "courthouse", "name": "courthouse"},
            {"id": "ignatius", "name": "ignatius"},
            {"id": "meetingroom", "name": "meetingroom"},
            {"id": "truck", "name": "truck"},
        ]
    }
})
