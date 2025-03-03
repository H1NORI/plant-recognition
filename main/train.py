
if __name__ == '__main__':
    from fastai.vision.all import *

    path = Path('main/datasets/plants')

    # print('Path exist? -', path.exists())
    #
    # print('Number of photos -', len(get_image_files(path)))


    plants = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128)
    )

    plants = plants.new(
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms()
    )

    dls = plants.dataloaders(path)
    # dls = plants.dataloaders(path, num_workers=0)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(6)

    learn.export()