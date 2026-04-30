"""Segmentation-aware tiling: tile images + Labelme JSON polygon annotations jointly."""
import os
import json
import cv2
import numpy as np
from utils.helpers import ensure_dir, get_image_files


def _create_polygon_mask(shape, img_h, img_w):
    """Create a binary mask from a Labelme polygon shape."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    points = np.array(shape["points"], dtype=np.float32).reshape((-1, 1, 2))
    points = points.astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def _extract_shapes_from_mask(mask, label, x_offset, y_offset, shape_type="polygon"):
    """Extract contours from a binary mask and convert to Labelme shapes."""
    shapes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) < 3:
            continue
        # Simplify the contour (optional, reduces point count)
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Convert to list of [x, y] points (already in local tile coordinates)
        points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
        if len(points) < 3:
            continue
        shapes.append({
            "label": label,
            "points": points,
            "shape_type": shape_type,
            "flags": {},
            "group_id": None,
            "description": "",
            "difficult": False,
            "attributes": {},
            "kie_linking": []
        })
    return shapes


def tile_segmentation_dataset(image_dir, ann_dir, output_dir, tile_w=256, tile_h=256,
                               overlap=0, discard_empty=False, discard_incomplete=False,
                               progress_callback=None):
    """
    Tile images + Labelme JSON annotations together.

    Args:
        image_dir: Directory containing source images
        ann_dir: Directory containing Labelme JSON annotations
        output_dir: Output root directory
        tile_w, tile_h: Tile dimensions
        overlap: Overlap pixels between tiles
        discard_empty: Skip tiles with no annotations
        discard_incomplete: Skip edge tiles smaller than tile_w x tile_h

    Returns:
        dict with statistics
    """
    out_img_dir = os.path.join(output_dir, "images")
    out_ann_dir = os.path.join(output_dir, "annotations")
    ensure_dir(out_img_dir)
    ensure_dir(out_ann_dir)

    # Find paired files
    img_files = {os.path.splitext(os.path.basename(f))[0]: f
                 for f in get_image_files(image_dir)}
    ann_files = {}
    for f in os.listdir(ann_dir):
        if f.endswith(".json"):
            ann_files[os.path.splitext(f)[0]] = os.path.join(ann_dir, f)

    # Find pairs (same basename)
    pairs = []
    for base, img_path in img_files.items():
        if base in ann_files:
            pairs.append((base, img_path, ann_files[base]))

    total_tiles = 0
    total_skipped_empty = 0
    total_skipped_incomplete = 0

    for idx, (base, img_path, ann_path) in enumerate(pairs):
        img = cv2.imread(img_path)
        if img is None:
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        ih, iw = img.shape[:2]
        shapes = ann_data.get("shapes", [])
        step_y = tile_h - overlap
        step_x = tile_w - overlap

        tile_idx = 0
        y = 0
        while y < ih:
            x = 0
            while x < iw:
                # Tile boundaries
                x2 = min(x + tile_w, iw)
                y2 = min(y + tile_h, ih)
                tw = x2 - x
                th = y2 - y

                # Handle incomplete tiles
                if tw < tile_w or th < tile_h:
                    if discard_incomplete:
                        total_skipped_incomplete += 1
                        if x + tile_w >= iw:
                            break
                        x += step_x
                        continue
                    # Pad the image tile
                    tile_img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                    tile_img[:th, :tw] = img[y:y2, x:x2]
                else:
                    tile_img = img[y:y2, x:x2].copy()

                # Extract shapes within this tile
                tile_shapes = []
                for shape in shapes:
                    label = shape.get("label", "")
                    # Create mask for this shape on the full image
                    mask = _create_polygon_mask(shape, ih, iw)
                    # Crop to tile region
                    tile_mask = mask[y:y2, x:x2]
                    if discard_incomplete and (tw < tile_w or th < tile_h):
                        # For incomplete tiles, pad the mask too
                        padded_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
                        padded_mask[:th, :tw] = tile_mask
                        tile_mask = padded_mask

                    if np.any(tile_mask):
                        # Extract shapes from the cropped/padded mask
                        extracted = _extract_shapes_from_mask(
                            tile_mask, label, x, y,
                            shape.get("shape_type", "polygon")
                        )
                        tile_shapes.extend(extracted)

                # Decide whether to save
                if discard_empty and not tile_shapes:
                    total_skipped_empty += 1
                else:
                    tile_name = f"{base}_x{x:04d}_y{y:04d}_w{tw}_h{th}"
                    # Save image
                    img_out = os.path.join(out_img_dir, f"{tile_name}.png")
                    cv2.imwrite(img_out, tile_img)

                    # Save annotation
                    tile_ann = {
                        "version": ann_data.get("version", "5.0.0"),
                        "flags": {},
                        "shapes": tile_shapes,
                        "imagePath": f"{tile_name}.png",
                        "imageData": None,
                        "imageHeight": tile_h if (discard_incomplete or (th >= tile_h)) else th,
                        "imageWidth": tile_w if (discard_incomplete or (tw >= tile_w)) else tw,
                        "description": f"tile from {base} at ({x},{y})"
                    }
                    ann_out = os.path.join(out_ann_dir, f"{tile_name}.json")
                    with open(ann_out, "w", encoding="utf-8") as f:
                        json.dump(tile_ann, f, indent=2, ensure_ascii=False)

                    total_tiles += 1
                tile_idx += 1

                if x + tile_w >= iw:
                    break
                x += step_x

            if y + tile_h >= ih:
                break
            y += step_y

        if progress_callback:
            progress_callback(idx + 1, len(pairs))

    return {
        "total_pairs": len(pairs),
        "total_tiles": total_tiles,
        "skipped_empty": total_skipped_empty,
        "skipped_incomplete": total_skipped_incomplete,
        "output_image_dir": out_img_dir,
        "output_ann_dir": out_ann_dir
    }


def tile_segmentation_single(image_path, ann_path, output_dir, tile_w=256, tile_h=256,
                              overlap=0, discard_empty=False, discard_incomplete=False):
    """Tile a single image + annotation pair (for single-image preview flow)."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    import tempfile, shutil
    tmp_img = os.path.join(tempfile.mkdtemp(), base + os.path.splitext(image_path)[1])
    tmp_ann = os.path.join(os.path.dirname(tmp_img), base + ".json")
    shutil.copy2(image_path, tmp_img)
    shutil.copy2(ann_path, tmp_ann)

    result = tile_segmentation_dataset(
        os.path.dirname(tmp_img), os.path.dirname(tmp_ann),
        output_dir, tile_w, tile_h, overlap,
        discard_empty, discard_incomplete
    )
    shutil.rmtree(os.path.dirname(tmp_img))
    return result
