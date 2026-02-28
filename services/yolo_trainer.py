"""YOLO format export and fine-tuning utilities."""

import os
import shutil
import yaml
from models import db
from models.dataset import Dataset
from models.image import Image
from models.label import Label
from services.image_service import get_image_path


def export_yolo_dataset(dataset_id: int, output_dir: str) -> str:
    """Export labeled dataset in YOLO detection format.

    Each digit character in ground_truth gets its own bounding box,
    subdivided equally across the ROI width.

    Returns path to the data.yaml file.
    """
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        raise ValueError(f'Dataset {dataset_id} not found')

    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    labeled = (
        db.session.query(Image, Label)
        .join(Label)
        .filter(Image.dataset_id == dataset_id)
        .all()
    )

    for img, label in labeled:
        # Copy image
        src = get_image_path(img)
        dst = os.path.join(images_dir, img.filepath)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        # Create YOLO label file
        label_file = os.path.join(labels_dir,
                                  os.path.splitext(img.filepath)[0] + '.txt')
        gt = label.ground_truth
        num = len(gt)
        if num == 0:
            continue

        digit_width = label.roi_width / num
        lines = []
        for i, ch in enumerate(gt):
            if not ch.isdigit():
                continue
            cls_id = int(ch)
            # YOLO format: class x_center y_center width height (normalized)
            cx = (label.roi_x + digit_width * (i + 0.5)) / img.width
            cy = (label.roi_y + label.roi_height / 2) / img.height
            w = digit_width / img.width
            h = label.roi_height / img.height
            lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')

        with open(label_file, 'w') as f:
            f.write('\n'.join(lines))

    # Write data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images',
        'val': 'images',
        'names': {i: str(i) for i in range(10)},
    }
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    return yaml_path
