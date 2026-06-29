from torch.utils.data import Dataset, Sampler
import torch
import numpy as np
import pandas as pd
import monai.transforms as T

class VertSegMap(T.MapTransform):
    def __init__(self, keys, vert_msk_key='vert_msk', subreg_msk_key='spine_msk', msk_key='vert_matched', output_msk_suffix='_seg_msk', allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys)

        self.keys = keys

        self.vert_msk_key = vert_msk_key
        self.subreg_msk_key = subreg_msk_key
        self.msk_key = msk_key
        self.output_msk_suffix = output_msk_suffix

    def __call__(self, data):
        vert_msk_label = data[self.msk_key]

        # Perform label extraction directly
        vert_msk = data[self.vert_msk_key]
        vert_msk[vert_msk != vert_msk_label] = 0
        vert_msk[vert_msk == vert_msk_label] = 1

        # Extract subregion mask
        subreg_msk = data[self.subreg_msk_key]
        subreg_msk[(subreg_msk != 49) & (subreg_msk != 50)] = 0
        subreg_msk[(subreg_msk == 49) | (subreg_msk == 50)] = 1

        # Combine masks and add them to the data dictionary
        for key in self.keys:
            result_msk = vert_msk * subreg_msk
            data[key + self.output_msk_suffix] = result_msk
            
        return data

class StackSegmentationToImage(T.MapTransform):
    """
    Custom MONAI transform to add the segmentation mask as a second channel to the image tensor.
    This transform modifies the "image" key by stacking the "seg_msk" tensor along the channel dimension.

    Args:
        image_key (str): Key for the image tensor.
        seg_key (str): Key for the segmentation mask tensor.
    """

    def __init__(self, image_key="image", seg_key="seg_msk"):
        super().__init__(keys=[image_key, seg_key])
        self.image_key = image_key
        self.seg_key = seg_key

    def __call__(self, data):
        d = dict(data)
        image = d[self.image_key]  # Shape: (1, H, W) or (C, H, W, D) for 3D
        seg_msk = d[self.seg_key]  # Shape: (1, H, W) or (1, H, W, D)

        if image.shape[1:] != seg_msk.shape[1:]:
            raise ValueError(f"Shape mismatch: Image shape {image.shape} and Seg Mask shape {seg_msk.shape} do not match.")

        # Stack along the channel dimension
        d[self.image_key] = torch.cat([image, seg_msk], dim=0)  # Now (2, H, W) or (2, H, W, D)

        return d


def create_spider_files(split="training"):
    meta = pd.read_csv("/DATA/NAS/datasets_source/mri_spine/dataset-spider/overview.csv")
    labels = pd.read_csv("/DATA/NAS/datasets_source/mri_spine/dataset-spider/radiological_gradings.csv")

    meta_train = meta[(meta.subset == split) & (meta.new_file_name.str.endswith("t2"))]

    img_path = "/DATA/NAS/datasets_source/mri_spine/dataset-spider/images/"
    mask_path = "/DATA/NAS/datasets_source/mri_spine/dataset-spider/masks/"

    files = list()
    for f in list(meta_train.new_file_name):
        rec = dict()
        rec['image'] = img_path + f + ".mha"
        rec['mask'] = mask_path + f + ".mha"
        rec['patient'] = int(f.split('_')[0])

        num_vertebrae = meta_train[meta_train.new_file_name == f]['num_vertebrae'].item()
        for v in range(1, num_vertebrae + 1):
            rec['ivd_label'] = v
            match = labels[(labels['Patient'] == rec['patient']) & (labels['IVD label']== v)]
            if len(match['Pfirrman grade']) != 1:
                print("Missing grade, skipping IVD")
                continue
            rec['pfirrman_grade'] = match['Pfirrman grade'].item() - 1
            files.append(rec.copy())

    # something wrong with this
    files = [f for f in files if not (f['patient'] == 256 and f['ivd_label']) == 8]
    
    return files


class CropMaskByLabel(T.Transform):
    def __init__(self, mask_key='mask', label_key='label', label_lambda_func=lambda x:x):
        super().__init__()
        self.mask_key = mask_key
        self.label_key = label_key
        self.label_lambda_func = label_lambda_func

    def __call__(self, data):
        d = dict(data)

        mask = d[self.mask_key]
        label = self.label_lambda_func(d[self.label_key])
        d[self.mask_key] = (mask == label).astype(mask.dtype)

        assert d[self.mask_key].sum(), "patient %d, label %d" % (d['patient'], label)
        
        return d

class AssertEmptyImaged(T.Transform):
    def __init__(self):
        super().__init__()
    def __call__(self, data):
        d = dict(data)

        assert all(dim > 0 for dim in d["image"].size()), \
            "File %d, Vert %d, %s, %s" % (d['PatientID'], d['Vert_idx'], str(d["image"].size()), str(d["image_seg_msk"].size()))

class OneExamPerPatientSampler(Sampler):
    """
    At each epoch, randomly selects one exam (date) per patient,
    then yields all row indices belonging to those selected exams.
    """
    def __init__(self, df, patient_col='patient_id', exam_col='date'):
        self.patient_to_exams = {}
        for patient, group in df.groupby(patient_col):
            exams = {}
            for exam, exam_group in group.groupby(exam_col):
                exams[exam] = exam_group.index.tolist()
            self.patient_to_exams[patient] = exams

    def __iter__(self):
        indices = []
        for patient, exams in self.patient_to_exams.items():
            chosen_exam = np.random.choice(list(exams.keys()))
            indices.extend(exams[chosen_exam])
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return sum(
            len(next(iter(exams.values())))
            for exams in self.patient_to_exams.values()
        )


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]
