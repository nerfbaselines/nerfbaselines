from ..registry import register


register(name="tanksandtemples", spec={
    "load_dataset_function": ".tanksandtemples:load_tanksandtemples_dataset",
    "priority": 140,
    "download_dataset_function": ".tanksandtemples:download_tanksandtemples_dataset",
})
