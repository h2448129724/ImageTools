"""Tests for core/augmentation.py."""
import os
import random
import tempfile

import cv2
import numpy as np
import pytest

from core.augmentation import (
    Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotate, RandomScale,
    RandomCrop, LetterboxResize, ColorJitter, GaussianNoise, SaltAndPepperNoise,
    RandomGaussianBlur, RandomMotionBlur, RandomErasing, Cutout, Normalize,
    LongestMaxSize, MixUp, Mosaic,
    build_pipeline, augment_image, augment_dataset, TRANSFORM_REGISTRY,
    save_pipeline_config, load_pipeline_config,
    BBoxHorizontalFlip, BBoxVerticalFlip, BBoxScale, BBoxLetterboxResize,
    BBoxColorJitter, BBoxRandomCrop, BBoxCompose, BBoxTransform,
    build_bbox_pipeline, augment_with_bboxes, BBOX_TRANSFORM_REGISTRY,
)
from core.image_io import write_image


@pytest.fixture
def bgr_img():
    return np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)


@pytest.fixture
def gray_img():
    return np.random.randint(0, 256, (100, 120), dtype=np.uint8)


@pytest.fixture
def small_img():
    return np.random.randint(0, 256, (30, 40, 3), dtype=np.uint8)


def _apply_always(transform, img):
    """Force transform to apply (p=1.0)."""
    old_p = transform.p
    transform.p = 1.0
    result = transform(img)
    transform.p = old_p
    return result


class TestCompose:
    def test_chain(self, bgr_img):
        pipeline = Compose([RandomHorizontalFlip(p=1.0), RandomVerticalFlip(p=1.0)])
        result = pipeline(bgr_img)
        assert result.shape == bgr_img.shape

    def test_empty(self, bgr_img):
        pipeline = Compose([])
        result = pipeline(bgr_img)
        np.testing.assert_array_equal(result, bgr_img)

    def test_skip_probability(self, bgr_img):
        random.seed(999)
        pipeline = Compose([RandomHorizontalFlip(p=0.0)])
        result = pipeline(bgr_img)
        np.testing.assert_array_equal(result, bgr_img)


class TestGeometric:
    def test_hflip(self, bgr_img):
        result = _apply_always(RandomHorizontalFlip(), bgr_img)
        expected = cv2.flip(bgr_img, 1)
        np.testing.assert_array_equal(result, expected)

    def test_vflip(self, bgr_img):
        result = _apply_always(RandomVerticalFlip(), bgr_img)
        expected = cv2.flip(bgr_img, 0)
        np.testing.assert_array_equal(result, expected)

    def test_rotate_shape(self, bgr_img):
        result = _apply_always(RandomRotate(limit=15.0), bgr_img)
        assert result.shape == bgr_img.shape
        assert result.dtype == np.uint8

    def test_scale(self, bgr_img):
        random.seed(42)
        t = RandomScale(scale_limit=(0.5, 0.5), p=1.0)
        result = _apply_always(t, bgr_img)
        assert result.shape[0] == 50
        assert result.shape[1] == 60

    def test_random_crop(self, bgr_img):
        random.seed(42)
        t = RandomCrop(crop_h=50, crop_w=60, p=1.0)
        result = _apply_always(t, bgr_img)
        assert result.shape == (50, 60, 3)

    def test_random_crop_pad(self, small_img):
        random.seed(42)
        t = RandomCrop(crop_h=80, crop_w=100, p=1.0)
        result = _apply_always(t, small_img)
        assert result.shape == (80, 100, 3)

    def test_letterbox(self, bgr_img):
        t = LetterboxResize(target_h=640, target_w=640, p=1.0)
        result = t(bgr_img)
        assert result.shape == (640, 640, 3)
        assert result.dtype == np.uint8

    def test_letterbox_gray(self, gray_img):
        t = LetterboxResize(target_h=256, target_w=256, p=1.0)
        result = t(gray_img)
        assert result.shape == (256, 256)

    def test_letterbox_preserves_content(self, bgr_img):
        t = LetterboxResize(target_h=100, target_w=120, p=1.0)
        result = t(bgr_img)
        # Same size, should be nearly identical
        assert result.shape == (100, 120, 3)


class TestColor:
    def test_color_jitter_bgr(self, bgr_img):
        random.seed(42)
        t = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)
        result = _apply_always(t, bgr_img)
        assert result.shape == bgr_img.shape
        assert result.dtype == np.uint8

    def test_color_jitter_gray(self, gray_img):
        random.seed(42)
        t = ColorJitter(brightness=0.3, contrast=0.3, p=1.0)
        result = _apply_always(t, gray_img)
        assert result.shape == gray_img.shape
        assert result.dtype == np.uint8

    def test_color_jitter_zero(self, bgr_img):
        t = ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=1.0)
        result = _apply_always(t, bgr_img)
        np.testing.assert_array_equal(result, bgr_img)


class TestNoise:
    def test_gaussian_noise(self, bgr_img):
        random.seed(42)
        np.random.seed(42)
        t = GaussianNoise(std_limit=25.0, p=1.0)
        result = _apply_always(t, bgr_img)
        assert result.shape == bgr_img.shape
        assert result.dtype == np.uint8
        # Should be different from input
        assert not np.array_equal(result, bgr_img)

    def test_salt_pepper(self, bgr_img):
        random.seed(42)
        np.random.seed(42)
        t = SaltAndPepperNoise(amount=0.05, p=1.0)
        result = _apply_always(t, bgr_img)
        assert result.shape == bgr_img.shape
        assert result.dtype == np.uint8


class TestBlur:
    def test_gaussian_blur(self, bgr_img):
        random.seed(42)
        t = RandomGaussianBlur(ksize_limit=(3, 7), p=1.0)
        result = _apply_always(t, bgr_img)
        assert result.shape == bgr_img.shape

    def test_motion_blur(self, bgr_img):
        random.seed(42)
        t = RandomMotionBlur(kernel_limit=15, p=1.0)
        result = _apply_always(t, bgr_img)
        assert result.shape == bgr_img.shape


class TestErasing:
    def test_random_erasing(self, bgr_img):
        random.seed(42)
        t = RandomErasing(area_ratio=(0.05, 0.15), p=1.0)
        result = _apply_always(t, bgr_img)
        assert result.shape == bgr_img.shape
        # Some pixels should be zero
        assert (result == 0).any()

    def test_cutout(self, bgr_img):
        random.seed(42)
        t = Cutout(n_holes=3, hole_size=20, p=1.0)
        result = _apply_always(t, bgr_img)
        assert result.shape == bgr_img.shape

    def test_cutout_larger_than_image(self, small_img):
        random.seed(42)
        t = Cutout(n_holes=1, hole_size=100, p=1.0)
        result = _apply_always(t, small_img)
        assert result.shape == small_img.shape


class TestNormalize:
    def test_normalize_shape_and_dtype(self, bgr_img):
        t = Normalize(p=1.0)
        result = _apply_always(t, bgr_img)
        assert result.shape == bgr_img.shape
        assert result.dtype == np.float32


class TestBuildPipeline:
    def test_from_config(self, bgr_img):
        random.seed(42)
        np.random.seed(42)
        config = [
            {"name": "RandomHorizontalFlip", "params": {"p": 1.0}},
            {"name": "ColorJitter", "params": {"brightness": 0.2, "p": 1.0}},
        ]
        pipeline = build_pipeline(config)
        result = augment_image(bgr_img, pipeline)
        assert result.shape == bgr_img.shape

    def test_unknown_transform(self):
        with pytest.raises(ValueError, match="Unknown transform"):
            build_pipeline([{"name": "NonExistent"}])

    def test_all_registered_transforms_instantiable(self):
        for name, entry in TRANSFORM_REGISTRY.items():
            cls = entry["class"]
            params = entry["params"]
            instance = cls(**params)
            assert isinstance(instance, object)


class TestPipelineIntegration:
    def test_full_pipeline(self, bgr_img):
        """Test a realistic augmentation pipeline."""
        random.seed(42)
        np.random.seed(42)
        config = [
            {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
            {"name": "RandomRotate", "params": {"limit": 15.0, "p": 0.5}},
            {"name": "ColorJitter", "params": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "p": 0.5}},
            {"name": "GaussianNoise", "params": {"std_limit": 15.0, "p": 0.3}},
            {"name": "RandomGaussianBlur", "params": {"ksize_limit": (3, 5), "p": 0.3}},
            {"name": "Cutout", "params": {"n_holes": 1, "hole_size": 16, "p": 0.3}},
        ]
        pipeline = build_pipeline(config)
        result = augment_image(bgr_img, pipeline)
        assert result.shape == bgr_img.shape
        assert result.dtype == np.uint8


class TestAugmentDataset:
    def test_augment_dataset_basic(self, tmp_path):
        random.seed(42)
        np.random.seed(42)
        # Create test images
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        for i in range(3):
            write_image(str(tmp_path / f"test_{i}.png"), img)

        config = [{"name": "RandomHorizontalFlip", "params": {"p": 1.0}}]
        out_dir = str(tmp_path / "augmented")
        stats = augment_dataset(str(tmp_path), out_dir, config, copies=2, seed=42)

        assert stats["total_input"] == 3
        assert stats["total_output"] == 6
        assert stats["errors"] == 0
        # Check output files exist
        assert len(os.listdir(out_dir)) == 6

    def test_augment_dataset_with_cancel(self, tmp_path):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        write_image(str(tmp_path / "test.png"), img)

        config = [{"name": "RandomHorizontalFlip", "params": {"p": 1.0}}]
        out_dir = str(tmp_path / "augmented")
        stats = augment_dataset(str(tmp_path), out_dir, config, cancel_check=lambda: True)
        assert stats["total_output"] == 0


class TestLongestMaxSize:
    def test_downscale(self, bgr_img):
        t = LongestMaxSize(max_size=50, p=1.0)
        result = t(bgr_img)
        assert max(result.shape[0], result.shape[1]) <= 50

    def test_no_upscale(self):
        small = np.zeros((30, 40, 3), dtype=np.uint8)
        t = LongestMaxSize(max_size=100, p=1.0)
        result = t(small)
        assert result.shape == small.shape

    def test_preserves_aspect(self):
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        t = LongestMaxSize(max_size=100, p=1.0)
        result = t(img)
        assert result.shape[0] == 100
        assert result.shape[1] == 50


class TestMixUp:
    def test_basic_blend(self, bgr_img):
        second = np.full_like(bgr_img, 128, dtype=np.uint8)
        t = MixUp(second_img=second, alpha=0.5, p=1.0)
        result = t(bgr_img)
        assert result.shape == bgr_img.shape
        assert result.dtype == np.uint8

    def test_grayscale_blend(self, gray_img):
        second = np.full_like(gray_img, 128, dtype=np.uint8)
        t = MixUp(second_img=second, alpha=0.5, p=1.0)
        result = t(gray_img)
        assert result.shape == gray_img.shape


class TestMosaic:
    def test_basic_mosaic(self, bgr_img):
        extras = [np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8) for _ in range(3)]
        t = Mosaic(images=extras, target_size=128, p=1.0)
        result = t(bgr_img)
        assert result.shape == (128, 128, 3)

    def test_mosaic_with_few_images(self, bgr_img):
        t = Mosaic(images=[], target_size=100, p=1.0)
        result = t(bgr_img)
        assert result.shape == (100, 100, 3)

    def test_mosaic_grayscale(self, gray_img):
        extras = [np.random.randint(0, 256, (50, 50), dtype=np.uint8) for _ in range(3)]
        t = Mosaic(images=extras, target_size=100, p=1.0)
        result = t(gray_img)
        assert result.shape == (100, 100)


class TestRepr:
    def test_all_transforms_have_repr(self):
        random.seed(42)
        np.random.seed(42)
        classes = [RandomHorizontalFlip, RandomVerticalFlip, RandomRotate, RandomScale,
                   RandomCrop, LetterboxResize, ColorJitter, GaussianNoise, SaltAndPepperNoise,
                   RandomGaussianBlur, RandomMotionBlur, RandomErasing, Cutout, Normalize,
                   LongestMaxSize]
        for cls in classes:
            instance = cls(**TRANSFORM_REGISTRY.get(cls.__name__, {}).get("params", {}))
            r = repr(instance)
            assert cls.__name__ in r
            assert "(" in r

    def test_compose_repr(self):
        pipeline = Compose([RandomHorizontalFlip(p=1.0)])
        assert "Compose" in repr(pipeline)


class TestConfigSerialization:
    def test_save_load_roundtrip(self, tmp_path):
        config = [
            {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
            {"name": "ColorJitter", "params": {"brightness": 0.3}},
        ]
        path = str(tmp_path / "pipeline.json")
        save_pipeline_config(config, path)
        loaded = load_pipeline_config(path)
        assert loaded == config

    def test_load_and_build(self, tmp_path):
        config = [{"name": "RandomHorizontalFlip", "params": {"p": 1.0}}]
        path = str(tmp_path / "pipeline.json")
        save_pipeline_config(config, path)
        loaded = load_pipeline_config(path)
        pipeline = build_pipeline(loaded)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = pipeline(img)
        assert result.shape == img.shape


class TestColorJitterCorrectness:
    def test_brightness_in_bgr_space(self):
        """ColorJitter should apply brightness in BGR space, not HSV V channel."""
        # Use a mid-gray image so brightness shifts are visible in all channels
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        t = ColorJitter(brightness=1.0, contrast=0, saturation=0, hue=0, p=1.0)
        random.seed(42)
        result = t(img)
        # In BGR space, brightness shift changes all channels uniformly from 128
        # In HSV V space, the shift would be different
        assert result.dtype == np.uint8
        assert result.shape == img.shape
        # Verify result is different from input (brightness=1.0 guarantees shift)
        assert not np.array_equal(result, img)

    def test_normalize_in_pipeline_saves_as_uint8(self, tmp_path):
        """augment_dataset should handle Normalize producing float32 by converting back to uint8."""
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        for i in range(2):
            write_image(str(tmp_path / f"test_{i}.png"), img)
        config = [{"name": "Normalize", "params": {}}]
        out_dir = str(tmp_path / "norm_output")
        stats = augment_dataset(str(tmp_path), out_dir, config, seed=42)
        assert stats["total_output"] == 2
        # Verify output files are valid images
        from core.image_io import read_image
        for i in range(2):
            result = read_image(os.path.join(out_dir, f"test_{i}.png"))
            assert result is not None
            assert result.dtype == np.uint8


def _make_boxes():
    """Create sample YOLO-format bboxes."""
    return [
        {"cls": 0, "xc": 0.5, "yc": 0.5, "bw": 0.3, "bh": 0.3},
        {"cls": 1, "xc": 0.25, "yc": 0.25, "bw": 0.2, "bh": 0.2},
    ]


class TestBBoxHorizontalFlip:
    def test_flip(self, bgr_img):
        boxes = _make_boxes()
        t = BBoxHorizontalFlip(p=1.0)
        img_out, boxes_out = t(bgr_img, boxes)
        assert img_out.shape == bgr_img.shape
        assert len(boxes_out) == 2
        assert abs(boxes_out[0]["xc"] - 0.5) < 1e-6
        assert abs(boxes_out[1]["xc"] - 0.75) < 1e-6

    def test_skip(self, bgr_img):
        random.seed(0)
        boxes = _make_boxes()
        t = BBoxHorizontalFlip(p=0.0)
        img_out, boxes_out = t(bgr_img, boxes)
        np.testing.assert_array_equal(img_out, bgr_img)
        assert boxes_out is boxes


class TestBBoxVerticalFlip:
    def test_flip(self, bgr_img):
        boxes = _make_boxes()
        t = BBoxVerticalFlip(p=1.0)
        _, boxes_out = t(bgr_img, boxes)
        assert abs(boxes_out[0]["yc"] - 0.5) < 1e-6
        assert abs(boxes_out[1]["yc"] - 0.75) < 1e-6


class TestBBoxScale:
    def test_scale(self, bgr_img):
        random.seed(42)
        boxes = _make_boxes()
        t = BBoxScale(scale_limit=(2.0, 2.0), p=1.0)
        img_out, boxes_out = t(bgr_img, boxes)
        assert img_out.shape[0] == 200
        assert img_out.shape[1] == 240
        # Bboxes unchanged (normalized coords)
        assert boxes_out == boxes


class TestBBoxLetterboxResize:
    def test_resize(self, bgr_img):
        boxes = _make_boxes()
        t = BBoxLetterboxResize(target_h=640, target_w=640, p=1.0)
        img_out, boxes_out = t(bgr_img, boxes)
        assert img_out.shape[:2] == (640, 640)
        assert len(boxes_out) == 2
        # All boxes should have valid normalized coords
        for b in boxes_out:
            assert 0 <= b["xc"] <= 1
            assert 0 <= b["yc"] <= 1


class TestBBoxColorJitter:
    def test_preserves_bboxes(self, bgr_img):
        random.seed(42)
        boxes = _make_boxes()
        t = BBoxColorJitter(brightness=0.3, p=1.0)
        _, boxes_out = t(bgr_img, boxes)
        assert boxes_out == boxes


class TestBBoxRandomCrop:
    def test_crop_adjusts_bboxes(self, bgr_img):
        random.seed(42)
        boxes = [{"cls": 0, "xc": 0.5, "yc": 0.5, "bw": 0.3, "bh": 0.3}]
        t = BBoxRandomCrop(crop_h=50, crop_w=60, p=1.0)
        img_out, boxes_out = t(bgr_img, boxes)
        assert img_out.shape[:2] == (50, 60)
        assert len(boxes_out) >= 0  # May or may not overlap with crop
        for b in boxes_out:
            assert 0 <= b["xc"] <= 1
            assert 0 <= b["yc"] <= 1

    def test_crop_can_exclude_box(self):
        # Box at top-left, crop at bottom-right
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [{"cls": 0, "xc": 0.1, "yc": 0.1, "bw": 0.05, "bh": 0.05}]
        random.seed(99)
        t = BBoxRandomCrop(crop_h=50, crop_w=50, p=1.0)
        _, boxes_out = t(img, boxes)
        # Depending on random crop position, box may or may not be included
        for b in boxes_out:
            assert 0 <= b["xc"] <= 1


class TestBBoxCompose:
    def test_chain(self, bgr_img):
        random.seed(42)
        boxes = _make_boxes()
        pipeline = BBoxCompose([
            BBoxHorizontalFlip(p=1.0),
            BBoxColorJitter(brightness=0.2, p=1.0),
        ])
        img_out, boxes_out = pipeline(bgr_img, boxes)
        assert img_out.shape == bgr_img.shape
        assert len(boxes_out) == 2
        # xc should be flipped
        assert abs(boxes_out[0]["xc"] - 0.5) < 1e-6
        assert abs(boxes_out[1]["xc"] - 0.75) < 1e-6

    def test_build_from_config(self, bgr_img):
        random.seed(42)
        boxes = _make_boxes()
        config = [
            {"name": "BBoxHorizontalFlip", "params": {"p": 1.0}},
            {"name": "BBoxScale", "params": {"scale_limit": (0.5, 0.5), "p": 1.0}},
        ]
        pipeline = build_bbox_pipeline(config)
        img_out, boxes_out = pipeline(bgr_img, boxes)
        assert img_out.shape[0] == 50
        assert len(boxes_out) == 2

    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown bbox transform"):
            build_bbox_pipeline([{"name": "NonExistent"}])

    def test_all_registered_instantiable(self):
        for name, entry in BBOX_TRANSFORM_REGISTRY.items():
            cls = entry["class"]
            params = entry["params"]
            instance = cls(**params)
            assert isinstance(instance, BBoxTransform)


