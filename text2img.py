import os
import config
from PIL import Image
from matplotlib import pyplot as plt


def main():
    text = input("word: ")
    text = text.strip().upper()

    length = len(text)
    plt.figure(figsize=(2.5 * length, 2.5))
    for i, el in enumerate(text):
        plt.subplot(1, length, i + 1)
        el_path = os.path.join(config.ALPHABET_IMAGES_PATH, f"{el}.{config.ALPHABET_IMAGES_EXTENSION}")
        with Image.open(el_path) as image:
            plt.imshow(image)
        plt.title(el)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()