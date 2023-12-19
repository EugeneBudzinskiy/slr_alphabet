import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import config


def load_label_decode_inverse_dict() -> list[str]:
    with open(config.TRAIN_LABEL_DECODE_JSON_PATH, mode="r", encoding="utf-8") as f:
        label_decode_dict = json.load(f)
    result = [""] * len(label_decode_dict)
    for key, val in label_decode_dict.items():
        result[val] = key
    return result


def main():
    label_decode_inv = load_label_decode_inverse_dict()
    model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)

    stream = cv2.VideoCapture(0)

    frame_step = stream.get(cv2.CAP_PROP_FPS) // config.N_FRAMES_PER_SECOND
    frame_index = 0
    while stream.isOpened():
        has_next, frame = stream.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break

        if has_next:
            if (frame_index + 1) % frame_step == 0:
                frame_index = 0

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode="RGB")
                width, height = image.size
                ms = min(width, height)
                x_off, y_off = (width - ms) // 2, (height - ms) // 2
                image = image.crop(box=(x_off, y_off, x_off + ms, y_off + ms))

                # noinspection PyTypeChecker
                nn_frame = np.asarray(image)
                cv2.imshow('NN view', cv2.cvtColor(nn_frame, cv2.COLOR_RGB2BGR))

                image = image.resize(size=(config.IMAGE_SIZE, config.IMAGE_SIZE), resample=3)

                # noinspection PyTypeChecker
                data = np.expand_dims(np.asarray(image, dtype="uint8"), axis=0)

                prediction = np.argmax(model.predict(data, verbose=0), axis=-1)
                print(label_decode_inv[int(prediction)])
            frame_index += 1
        else:
            break
    stream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
