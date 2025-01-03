from pathlib import Path


from fastbook import *
from fastai.vision.augment import Resize
from fastai.vision.data import ImageBlock
from fastai.vision.learner import vision_learner
from torchvision.models.quantization import resnet50, resnext101_32x8d


SCRIPT_DIR = Path(__file__).parent

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

base_dir = Path("/Users/lukaswoodtli/Meine Ablage/fhnw_computer_vision_mit_deep_learning_projekt/")
data_set_dir = base_dir / "data" / "data_set"
photos_data_set = data_set_dir / 'photos'
renders_data_set = data_set_dir / 'renders'

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.05, seed=42),
    get_y=parent_label,
    item_tfms=[
        Resize(256),
#        transforms.Normalize(mean=mean, std=std),
#transforms.ToTensor()
    ],
    batch_tfms = aug_transforms
).dataloaders(photos_data_set)

learn = vision_learner(dls, resnext101_32x8d, metrics=error_rate)
learn.fine_tune(3)
