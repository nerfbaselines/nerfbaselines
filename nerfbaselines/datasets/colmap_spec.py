from nerfbaselines import register


register({
    "id": "colmap",
    "load_dataset_function": "nerfbaselines.datasets.colmap:load_colmap_dataset",
})
