import logging
import os
import json
from nerfbaselines import UnloadedDataset, DatasetNotFoundError
from .colmap import load_colmap_dataset


def extract_train_test_split(original_path):
    with open(os.path.join(original_path, 'transforms.json'), 'r') as f:
        transforms_data = json.load(f)
        strip_prefix = lambda x: x.split('/')[-1]  # noqa
        image_names = [strip_prefix(x["file_path"]) for x in transforms_data["frames"]]
        image_names = sorted(image_names)
    with open(os.path.join(original_path, "split.json"), 'r') as f:
        split_data = json.load(f)
    train_list = [image_names[x] for x in split_data['clutter']]
    test_list = [image_names[x] for x in split_data['extra']]
    assert len(train_list) > 0, 'No training images found'
    assert len(test_list) > 0, 'No test images found'
    return train_list, test_list


def _get_downscale_factor(path):
    scale = 4
    lastpart = os.path.basename(os.path.abspath(path))
    if "patio" in lastpart and "high" not in lastpart:
        scale = 2
    if "arc" in lastpart and "triomphe" in lastpart:
        scale = 2
    return scale


def preprocess_nerfonthego_dataset(path, output):
    os.makedirs(os.path.join(output, 'original'), exist_ok=True)

    # First, we create the train/test splits
    train_list, test_list = extract_train_test_split(path)
    with open(os.path.join(output, 'train_list.txt'), 'w') as f:
        for item in train_list:
            f.write("%s\n" % item)
    with open(os.path.join(output, 'test_list.txt'), 'w') as f:
        for item in test_list:
            f.write("%s\n" % item)

    # Next, we run COLMAP to extract the features and matches
    logging.info("Running feature extractor")
    os.system(r"""colmap feature_extractor \
                    --database_path {shutil.quote(os.path.join(output, 'original', 'database.db'))} \ 
                    --image_path {shutil.quote(os.path.join(path, 'original', 'images'))} \
                    --ImageReader.camera_model SIMPLE_RADIAL \
                    --ImageReader.single_camera 1 \
                    --SiftExtraction.use_gpu 1""")

    logging.info("Running exhaustive feature matcher")
    os.system(r"""colmap exhaustive_matcher \
                    --database_path {shutil.quote(os.path.join(output, 'original', 'database.db'))} \
                    --SiftMatching.use_gpu 1""")

    logging.info("Running mapper")
    os.system(r"""colmap mapper \
                    --database_path {shutil.quote(os.path.join(output, 'original', 'database.db'))} \
                    --image_path {shutil.quote(os.path.join(path, 'original', 'images'))} \
                    --output_path {shutil.quote(os.path.join(output, 'original', 'sparse'))}""")

    # Undistort images
    logging.info("Undistorting images")
    os.system(r"""colmap image_undistorter \
                    --image_path {shutil.quote(os.path.join(path, 'original', 'images'))} \
                    --input_path {shutil.quote(os.path.join(output, 'original', 'sparse', '0'))} \
                    --output_path {shutil.quote(os.path.join(output))} \
                    --output_type COLMAP \
                    --max_image_size 4000""")

    # Downsize images
    scale = _get_downscale_factor(path)
    logging.info(f"Downsizing images by factor of {scale}")

    os.makedirs(os.path.join(output, f'images_{scale}'), exist_ok=True)
    for image in os.listdir(os.path.join(output, 'images')):
        os.system(f"convert {os.path.join(output, 'images', image)} -resize {100 / scale}% -quality 100 {os.path.join(output, f'images_{scale}', image)}")

    # Save metadata
    metadata = {
        "id": "nerfonthego",
        "downscale_factor": scale,
        "scene": os.path.split(path)[-1].replace("_", "-")
    }
    with open(os.path.join(output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)


def load_nerfonthego_dataset(path: str, split: str, **kwargs) -> UnloadedDataset:
    is_onthego = False
    if os.path.exists("metadata.json"):
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
            is_onthego = metadata.get("id") == "nerfonthego"
    if not is_onthego:
        raise DatasetNotFoundError("This dataset is not an nerfonthego dataset. The folder is mmissing a metadata.json file with the name field set to 'nerfonthego'")
    assert split in ['train', 'test']

    scale = _get_downscale_factor(path)
    dataset = load_colmap_dataset(path, split=split, 
                                  images_path=f"images_{scale}",
                                  **kwargs)
    metadata = dataset["metadata"]
    metadata["id"] = "nerfonthego"
    metadata["downscale_factor"] = scale
    metadata["scene"] = os.path.split(path)[-1].replace("_", "-")
    return dataset


__all__ = ["load_nerfonthego_dataset"]
