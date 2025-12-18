import numpy as np
import sys
sys.path.append("C:/Users/proks/OneDrive/Documents/GitHub/2025_Hackathon")
from generate_synthetic_training_data_TP import generate_stem_image_with_continuous_gb, rotate_flip_image_and_bboxes
from pathlib import Path
from PIL import Image
from tqdm import tqdm

training_data_dir = r"C:\Users\proks\OneDrive\Documents\GitHub\2025_Hackathon\generated artificial data v2".replace("\\", "/")
training_data_dir = Path(training_data_dir)
# training_data_dir = Path("./training_data")
img_size = 768

conditions = [
    {
        'seed': 1,
        # 'interstitial': 4,
    },
    {
        'seed': 2,
        # 'interstitial': 6,
    },
    {
        'seed': 3,
        # 'interstitial': 8,
    },
]

if __name__ == "__main__":
    training_data_dir.mkdir(exist_ok=True)

    images_dir = training_data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    images_validation_dir = images_dir / "validation"
    images_validation_dir.mkdir(exist_ok=True)
    images_train_dir = images_dir / "train"
    images_train_dir.mkdir(exist_ok=True)


    labels_dir = training_data_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    labels_validation_dir = labels_dir / "validation"
    labels_validation_dir.mkdir(exist_ok=True)
    labels_train_dir = labels_dir / "train"
    labels_train_dir.mkdir(exist_ok=True)

    rotations = [0, 90, 180, 270]
    reflections = ['none', 'horizontal', 'vertical', 'both']

    number_of_images = len(conditions) * len(rotations) * len(reflections)
    print(f"Generating {number_of_images} images...")

    training_split = 0.8

    # define random indices to be in the trianing set
    all_indices = np.arange(number_of_images)
    np.random.seed(0)
    np.random.shuffle(all_indices)
    split_index = int(training_split * number_of_images)
    training_indices = set(all_indices[:split_index])

    index = 0
    for condition_index, condition in tqdm(enumerate(conditions)):
        # print(img_size)
        # #%%
        stem_image, bboxes = generate_stem_image_with_continuous_gb(
            img_size=img_size,
            **condition
        )
        pil_stem_image = Image.fromarray((stem_image * 255).astype(np.uint8))
        # Generate augmented images and save
        for rot in rotations:
            for refl in reflections:
                augmented_image, augmented_bboxes = rotate_flip_image_and_bboxes(
                    pil_stem_image,
                    bboxes,
                    rot,
                    refl
                )

                # Determine if this image is for training or validation
                if index in training_indices:
                    image_save_path = images_train_dir / f"image_cond{condition_index:04d}_rot{rot}_refl{refl}.tif"
                    label_save_path = labels_train_dir / f"image_cond{condition_index:04d}_rot{rot}_refl{refl}.txt"
                else:
                    image_save_path = images_validation_dir / f"image_cond{condition_index:04d}_rot{rot}_refl{refl}.tif"
                    label_save_path = labels_validation_dir / f"image_cond{condition_index:04d}_rot{rot}_refl{refl}.txt"

                # Save image
                augmented_image.save(image_save_path)

                # Save labels in YOLO format (class_id x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4)
                with open(label_save_path, 'w') as f:
                    for bbox in augmented_bboxes:
                        bbox_str = ' '.join([f"{coord:.6f}" for coord in bbox])
                        f.write(f"{bbox_str}\n")

                index += 1