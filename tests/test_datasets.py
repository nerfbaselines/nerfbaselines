import numpy as np
import json
import os
from unittest import mock
import contextlib
import pytest


def _test_generic_dataset(tmp_path, dataset_name, scene, train_size, test_size):
    from nerfbaselines.datasets import load_dataset

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch("nerfbaselines.datasets._common.NB_PREFIX", tmp_path))

        dataset = load_dataset(f"external://{dataset_name}/{scene}", 
                               split="train")
        assert dataset is not None
        assert dataset["metadata"].get("id") == dataset_name
        assert dataset["metadata"].get("scene") == scene
        assert dataset["images"] is not None
        assert len(dataset["cameras"]) == len(dataset["images"])
        assert dataset["image_paths_root"] is not None
        for i in range(len(dataset["images"])):
            w, h = dataset["cameras"][i].image_sizes
            assert tuple(dataset["images"][i].shape[:2]) == (h, w)
            assert dataset["image_paths"][i].startswith(dataset["image_paths_root"])
        assert "viewer_transform" in dataset["metadata"]
        assert "viewer_initial_pose" in dataset["metadata"]
        assert "expected_scene_scale" in dataset["metadata"]

        # Assert dataset was downloaded
        assert (tmp_path / "datasets" / dataset_name / scene).exists()
        assert (tmp_path / "datasets" / dataset_name / scene / "nb-info.json").exists()
        # Assert loader is set for the dataset
        with (tmp_path / "datasets" / dataset_name / scene / "nb-info.json").open("r", encoding="utf8") as fp:
            metadata = json.load(fp)
            assert metadata.get("loader") is not None, f"Loader not set for {dataset_name}/{scene}"

        # Lest load test split
        test_dataset = load_dataset(f"external://{dataset_name}/{scene}",
                                    split="test")
        assert test_dataset is not None
        assert test_dataset["metadata"].get("id") == dataset_name
        assert test_dataset["metadata"].get("scene") == scene
        assert test_dataset["images"] is not None
        assert len(test_dataset["cameras"]) == len(test_dataset["images"])
        assert test_dataset["image_paths_root"] is not None
        for i in range(len(test_dataset["images"])):
            w, h = test_dataset["cameras"][i].image_sizes
            assert tuple(test_dataset["images"][i].shape[:2]) == (h, w)
            assert test_dataset["image_paths"][i].startswith(test_dataset["image_paths_root"])

        _train_size = len(dataset["images"])
        _test_size = len(test_dataset["images"])
        # print([os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"]][:5])
        assert (_train_size, _test_size) == (train_size, test_size), f"Expected ({train_size}, {test_size}), got ({_train_size}, {_test_size})"

        # Assert no overlap between train and test sets
        intersection = set(dataset["image_paths"]) & set(test_dataset["image_paths"])
        assert len(intersection) == 0, f"Intersection between train and test sets: {intersection}"

    return dataset, test_dataset


@pytest.mark.dataset("blender")
def test_blender_dataset(tmp_path):
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "blender", "lego", train_size=100, test_size=200)
    assert train_dataset["metadata"].get("type") == "object-centric"
    assert test_dataset["metadata"].get("type") == "object-centric"
    assert train_dataset["metadata"].get("evaluation_protocol") == "nerf"
    assert test_dataset["metadata"].get("evaluation_protocol") == "nerf"
    test_images = [os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"][:5]]
    assert test_images == ['test/r_0.png', 'test/r_1.png', 'test/r_2.png', 'test/r_3.png', 'test/r_4.png']
    assert test_dataset["images"][0].shape == (800, 800, 4)
    np.testing.assert_allclose(
        test_dataset["cameras"][0].intrinsics,
        np.array([1111.1111, 1111.1111,  400.    ,  400.    ], dtype=np.float32)
    )


@pytest.mark.dataset("mipnerf360")
@pytest.mark.dataset("mipnerf360-sparse")
def test_mipnerf360_dataset(tmp_path):
    # Test mipnerf360
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "mipnerf360", "flowers", train_size=151, test_size=22)
    assert train_dataset["metadata"].get("type") == "object-centric"
    assert test_dataset["metadata"].get("type") == "object-centric"
    assert train_dataset["metadata"].get("evaluation_protocol") == "nerf"
    assert test_dataset["metadata"].get("evaluation_protocol") == "nerf"
    test_images = [os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"][:5]]
    assert test_images == ['_DSC9040.JPG', '_DSC9048.JPG', '_DSC9056.JPG', '_DSC9064.JPG', '_DSC9072.JPG']
    assert test_dataset["images"][0].shape == (828, 1256, 3)

    # Test mipnerf360-sparse
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "mipnerf360-sparse", "flowers-n12", train_size=12, test_size=22)
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "mipnerf360-sparse", "flowers-n24", train_size=24, test_size=22)
    assert train_dataset["metadata"].get("type") == "object-centric"
    assert test_dataset["metadata"].get("type") == "object-centric"
    assert train_dataset["metadata"].get("evaluation_protocol") == "nerf"
    assert test_dataset["metadata"].get("evaluation_protocol") == "nerf"
    test_images = [os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"][:5]]
    assert test_images == ['_DSC9040.JPG', '_DSC9048.JPG', '_DSC9056.JPG', '_DSC9064.JPG', '_DSC9072.JPG']
    assert test_dataset["images"][0].shape == (828, 1256, 3)
    np.testing.assert_allclose(
        test_dataset["cameras"][0].intrinsics,
        np.array([1081.9324, 1071.3643,  628.    ,  414.    ], dtype=np.float32)
    )


@pytest.mark.dataset("zipnerf")
def test_zipnerf_dataset(tmp_path):
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "zipnerf", "alameda", train_size=1517, test_size=217)
    assert train_dataset["metadata"].get("evaluation_protocol") == "nerf"
    assert test_dataset["metadata"].get("evaluation_protocol") == "nerf"
    test_images = [os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"][:5]]
    assert test_images == ['indoor_DSC06836.JPG', 'indoor_DSC07249.JPG', 'indoor_DSC08034.JPG', 'indoor_DSC08058.JPG', 'indoor_DSC06500.JPG']
    assert test_dataset["images"][0].shape == (793, 1394, 3)
    np.testing.assert_allclose(
        test_dataset["cameras"][0].intrinsics,
        np.array([603.87744, 604.83496, 697.     , 396.5    ], dtype=np.float32)
    )


@pytest.mark.dataset("hierarchical-3dgs")
def test_hierarchical_3dgs_dataset(tmp_path):
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "hierarchical-3dgs", "smallcity", train_size=1470, test_size=30)
    assert train_dataset["metadata"].get("evaluation_protocol") == "nerf"
    assert test_dataset["metadata"].get("evaluation_protocol") == "nerf"
    test_images = [os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"][:5]]
    assert test_images == ['pass2_0424.png', 'pass1_0213.png', 'pass2_1480.png', 'pass3_0036.png', 'pass1_0743.png']
    assert test_dataset["images"][0].shape == (690, 1024, 3)
    np.testing.assert_allclose(
        test_dataset["cameras"][0].intrinsics,
        np.array([486.9701 , 487.09015, 512.     , 345.     ], dtype=np.float32)
    )


@pytest.mark.dataset("llff")
def test_llff_dataset(tmp_path):
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "llff", "fern", train_size=17, test_size=3)
    assert train_dataset["metadata"].get("type") == "forward-facing"
    assert test_dataset["metadata"].get("type") == "forward-facing"
    assert train_dataset["metadata"].get("evaluation_protocol") == "nerf"
    assert test_dataset["metadata"].get("evaluation_protocol") == "nerf"
    test_images = [os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"][:5]]
    assert test_images == ['images_4/image000.png', 'images_4/image008.png', 'images_4/image016.png']
    assert test_dataset["images"][0].shape == (756, 1008, 3)
    np.testing.assert_allclose(
        test_dataset["cameras"][0].intrinsics,
        np.array([815.131583, 815.131583, 504.      , 378.      ], dtype=np.float32)
    )


@pytest.mark.dataset("phototourism")
def test_phototourism_dataset(tmp_path):
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "phototourism", "sacre-coeur", train_size=830, test_size=21)
    assert train_dataset["metadata"].get("evaluation_protocol") == "nerfw"
    assert test_dataset["metadata"].get("evaluation_protocol") == "nerfw"
    test_images = [os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"][:5]]
    assert test_images == ['25927611_9367586008.jpg', '25422767_2496685776.jpg', 
                           '24887636_5651358818.jpg', '24540065_12909555815.jpg', 
                           '24454809_14006921991.jpg']
    assert test_dataset["images"][0].shape == (1020, 680, 3)
    np.testing.assert_allclose(
        test_dataset["cameras"][0].intrinsics,
        np.array([1335.584, 1335.584,  340.   ,  510.   ], dtype=np.float32)
    )


@pytest.mark.dataset("tanksandtemples")
def test_tanksandtemples_dataset(tmp_path):
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "tanksandtemples", "truck", train_size=219, test_size=32)
    assert train_dataset["metadata"].get("evaluation_protocol") == "default"
    assert test_dataset["metadata"].get("evaluation_protocol") == "default"
    test_images = [os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"][:5]]
    assert test_images == ['000001.jpg', '000009.jpg',
                           '000017.jpg', '000025.jpg', '000033.jpg']
    assert test_dataset["images"][0].shape == (543, 979, 3)
    np.testing.assert_allclose(
        test_dataset["cameras"][0].intrinsics,
        np.array([579.3541, 579.8881, 489.5   , 271.5   ], dtype=np.float32)
    )


@pytest.mark.dataset("seathru-nerf")
def test_seathru_nerf_dataset(tmp_path):
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "seathru-nerf", "curasao", train_size=18, test_size=3)
    assert train_dataset["metadata"].get("type") == "forward-facing"
    assert test_dataset["metadata"].get("type") == "forward-facing"
    assert train_dataset["metadata"].get("evaluation_protocol") == "default"
    assert test_dataset["metadata"].get("evaluation_protocol") == "default"
    test_images = [os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"][:5]]
    assert test_images == ['MTN_1288.png', 'MTN_1296.png', 'MTN_1304.png']
    assert test_dataset["images"][0].shape == (1182, 1776, 3)
    np.testing.assert_allclose(
        test_dataset["cameras"][0].intrinsics,
        np.array([1960.0787 , 1961.7875 ,  894.37976,  580.6337 ], dtype=np.float32)
    )


@pytest.mark.dataset("nerfstudio")
def test_nerfstudio_dataset(tmp_path):
    train_dataset, test_dataset = _test_generic_dataset(
        tmp_path, "nerfstudio", "egypt", train_size=272, test_size=30)
    assert train_dataset["metadata"].get("evaluation_protocol") == "default"
    assert test_dataset["metadata"].get("evaluation_protocol") == "default"
    test_images = [os.path.relpath(x, test_dataset["image_paths_root"]) for x in test_dataset["image_paths"][:5]]
    assert test_images == ["frame_00011.png", "frame_00021.png", "frame_00031.png", "frame_00041.png", "frame_00051.png"]
    assert test_dataset["images"][0].shape == (540, 960, 3)
    np.testing.assert_allclose(
        test_dataset["cameras"][0].intrinsics,
        np.array([838.04047, 838.1078 , 480.7128 , 265.42245], dtype=np.float32)
    )


def test_phototourism_evaluation_protocol(tmp_path):
    from nerfbaselines import new_cameras, new_dataset
    from nerfbaselines.evaluation import render_all_images

    method = mock.MagicMock()
    method.render.return_value = {
        "color": np.zeros((50, 60, 3), dtype=np.uint8),
    }
    method.optimize_embedding.return_value = {
        "embedding": np.zeros(12, dtype=np.float32),
    }

    list(render_all_images(method, new_dataset(
        images=[np.zeros((50, 60, 3), dtype=np.uint8) for _ in range(2)],
        image_paths=[str(tmp_path / f"image_{i}.png") for i in range(2)],
        cameras=new_cameras(
            poses=np.eye(4)[None, ...].repeat(2, axis=0)[..., :3, :4],
            intrinsics=np.array([[50, 50, 30, 25], [50, 50, 30, 25]], dtype=np.float32),
            camera_models=np.array([0, 0], dtype=np.int32),
            image_sizes=np.array([[60, 50], [60, 50]], dtype=np.int32),
        ),
        metadata={
            "evaluation_protocol": "nerfw",
        }
    ), output=str(tmp_path / "output")))
    assert method.render.call_count == 2
    args, kwargs = method.render.call_args
    assert len(args) == 1
    args[0].item()  # Assert single camera
    assert kwargs.get("options") is not None
    assert kwargs["options"].get("embedding") is not None
    assert kwargs["options"]["embedding"].shape == (12,)

    assert method.optimize_embedding.call_count == 2
    args, kwargs = method.optimize_embedding.call_args
    assert len(args) == 1
    assert kwargs["embedding"] is None


def test_all_dataset_tested():
    from nerfbaselines import get_supported_datasets
    untested_datasets = set(get_supported_datasets())
    for v in globals().values():
        for mark in getattr(v, "pytestmark", []):
            if isinstance(mark, pytest.Mark) and mark.name == "dataset":
                dataset = mark.args[0]
                untested_datasets.discard(dataset)
    assert len(untested_datasets) == 0, f"Untested datasets: {untested_datasets}"
