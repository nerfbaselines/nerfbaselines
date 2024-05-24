from ..registry import register


register(name="bundler", spec={
    "load_dataset_function": ".bundler:load_bundler_dataset",
    "priority": 50,
})
