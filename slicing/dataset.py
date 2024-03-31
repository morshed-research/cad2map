import numpy as np
import albumentations as A

from pathlib import Path
from PIL import Image

"""
Class to hold images to slice over
"""
class DatasetAdaptor:
    def __init__(self, images_dir_path, annotations_dataframe):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.file_name_to_idx = {
            image_id: idx
            for idx, image_id in enumerate(self.annotations_df.file_name.unique())
        }
        self.idx_to_file_name = {v: k for k, v, in self.file_name_to_idx.items()}

    def __len__(self) -> int:
        return len(self.file_name_to_idx)

    def __getitem__(self, index):
        file_name = self.idx_to_file_name[index]
        image = Image.open(f"{self.images_dir_path}/{file_name}")
        return np.array(image)

"""
Class to hold images slices
"""
class ImageSliceDetectionDataset:
    def __init__(
        self,
        ds_adaptor,
        slices_df,
        as_slice=True,
        slice_height=250,
        slice_width=250,
        transforms=None,
        bbox_min_area=0.1,
        bbox_min_visibility=0.1,
    ):
        self.ds = ds_adaptor
        self.slices_df = slices_df
        self.as_slice = as_slice
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.transforms = transforms
        self.bbox_min_area = bbox_min_area
        self.bbox_min_visibility = bbox_min_visibility

    def __len__(self) -> int:
        return len(self.slices_df)

    def _apply_transforms(self, transform_list, image, boxes, classes):
        transforms = A.Compose(
            transform_list,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_visibility=self.bbox_min_visibility,
                min_area=self.bbox_min_area,
                label_fields=["labels"],
            ),
        )

        transformed = transforms(image=image, bboxes=boxes, labels=classes)

        image = transformed["image"]
        boxes = np.array(transformed["bboxes"])
        classes = np.array(transformed["labels"])

        return image, boxes, classes

    def create_deterministic_crop_transforms(self, slice_corners):
        return [
            A.Crop(*slice_corners),
            A.PadIfNeeded(self.slice_height, self.slice_width, border_mode=0),
        ]

    def __getitem__(self, index):
        row = self.slices_df.iloc[index]

        file_name = row.file_name
        adaptor_idx = self.ds.file_name_to_idx[file_name]

        image = self.ds[adaptor_idx]

        transforms = []

        if self.as_slice:
            slice_bbox = row[["xmin", "ymin", "xmax", "ymax"]].values
            transforms.extend(self.create_deterministic_crop_transforms(slice_bbox))

        if self.transforms:
            transforms.extend(self.transforms)

        image, bboxes, class_labels = self._apply_transforms(
            transforms, image, [], []
        )

        return image, list(row[["xmin", "ymin", "xmax", "ymax"]].values)