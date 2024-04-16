from ..registry import register


register(name="colmap", spec={
    "load_dataset_function": ".colmap:load_colmap_dataset",
    "priority": 100,
})
