from nerfbaselines import register


register({
    "id": "colmap",
    "load_dataset_function": ".colmap:load_colmap_dataset",
    "priority": 100,
})
