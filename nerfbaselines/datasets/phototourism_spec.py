from ..registry import register


register(name="phototourism", spec={
    "load_dataset_function": ".phototourism:load_phototourism_dataset",
    "priority": 170,
    "download_dataset_function": ".phototourism:download_phototourism_dataset",
    "evaluation_protocol": {
        "evaluation_protocol": ".phototourism:NerfWEvaluationProtocol",
        "name": "nerfw",
    }
})
