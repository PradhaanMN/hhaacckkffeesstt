# Dataset Directory

Place your COCO format dataset here following this structure:

```
dataset/
├── train/
│   ├── annotations.json
│   └── images/
│       ├── image001.jpg
│       ├── image002.jpg
│       └── ...
├── val/
│   ├── annotations.json
│   └── images/
│       ├── image001.jpg
│       └── ...
└── test/
    ├── annotations.json
    └── images/
        ├── image001.jpg
        └── ...
```

## COCO Format Reference

Your `annotations.json` should follow this structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image001.jpg",
      "width": 640,
      "height": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234.5,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "solar_panel",
      "supercategory": "object"
    }
  ]
}
```

## Notes

- `bbox` format is `[x, y, width, height]` where (x, y) is the top-left corner
- All coordinates are in pixels
- `image_id` in annotations must match an `id` in images
- `category_id` must match an `id` in categories

The training script will automatically:
- Validate your annotations
- Fix common issues (missing categories, invalid bboxes)
- Convert to YOLO format
- Create the necessary directory structure
