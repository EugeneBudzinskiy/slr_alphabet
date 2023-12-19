import os
import json
import numpy as np
from PIL import Image
import config


def process_raw_dataset(raw_dataset_path: str, data_memmap_path: str, label_npy_path: str, label_decode_json_path,
                        image_size: int, image_channel: int, n_images_per_folder: int, number_of_classes: int):
    run_idx = 0
    image_total_number = number_of_classes * n_images_per_folder

    memmap_shape = (image_total_number, image_size, image_size, image_channel)
    memmap_array = np.memmap(data_memmap_path, dtype="uint8", mode="w+", shape=memmap_shape)

    label_decode_dict = {}
    label = np.zeros(image_total_number, dtype="uint8")

    for folder in os.listdir(raw_dataset_path):
        src_path = os.path.join(raw_dataset_path, folder)
        print(f"Convert images from: {src_path}")

        if folder not in label_decode_dict:
            label_decode_dict[folder] = len(label_decode_dict)

        available_files = os.listdir(src_path)
        random_idx = np.arange(len(available_files))
        np.random.shuffle(random_idx)
        random_idx = random_idx[:n_images_per_folder]

        for idx in random_idx:
            label[run_idx] = label_decode_dict[folder]
            with Image.open(fp=f"{src_path}/{available_files[idx]}") as image:
                image = image.resize(size=(image_size, image_size), resample=3)
                # noinspection PyTypeChecker
                memmap_array[run_idx] = np.asarray(image, dtype="uint8")
                run_idx += 1

    np.save(file=label_npy_path, arr=label)
    with open(label_decode_json_path, mode="w", encoding="utf-8") as f:
        json.dump(label_decode_dict, fp=f, ensure_ascii=True, indent=4)


def main():
    # TRAIN dataset processing
    process_raw_dataset(
        raw_dataset_path=config.RAW_TRAIN_DATASET_PATH,
        data_memmap_path=config.TRAIN_DATA_MEMMAP_PATH,
        label_npy_path=config.TRAIN_LABEL_NPY_PATH,
        label_decode_json_path=config.TRAIN_LABEL_DECODE_JSON_PATH,
        image_size=config.IMAGE_SIZE,
        image_channel=config.IMAGE_CHANNEL,
        n_images_per_folder=config.TRAIN_N_IMAGES_PER_FOLDER,
        number_of_classes=config.NUMBER_OF_CLASSES
    )

    # TEST dataset processing
    process_raw_dataset(
        raw_dataset_path=config.RAW_TEST_DATASET_PATH,
        data_memmap_path=config.TEST_DATA_MEMMAP_PATH,
        label_npy_path=config.TEST_LABEL_NPY_PATH,
        label_decode_json_path=config.TEST_LABEL_DECODE_JSON_PATH,
        image_size=config.IMAGE_SIZE,
        image_channel=config.IMAGE_CHANNEL,
        n_images_per_folder=config.TEST_N_IMAGES_PER_FOLDER,
        number_of_classes=config.NUMBER_OF_CLASSES
    )


if __name__ == '__main__':
    main()
