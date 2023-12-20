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
    prediction = None

    stream = cv2.VideoCapture(0)
    frame_step = stream.get(cv2.CAP_PROP_FPS) // config.N_FRAMES_PER_SECOND
    frame_index = 0
    while stream.isOpened():
        has_next, frame = stream.read()
        frame = cv2.flip(frame, 1)
        if has_next:
            if (frame_index + 1) % frame_step == 0:
                frame_index = 0

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode="RGB")
                width, height = image.size
                ms = min(width, height)
                x_off, y_off = (width - ms) // 2, (height - ms) // 2
                image = image.crop(box=(x_off, y_off, x_off + ms, y_off + ms))
                image = image.resize(size=(config.IMAGE_SIZE, config.IMAGE_SIZE), resample=3)

                # noinspection PyTypeChecker
                data = np.expand_dims(np.asarray(image, dtype="uint8"), axis=0)
                prediction = np.argmax(model.predict(data, verbose=0))

            frame_index += 1
        else:
            break

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break

        text_result = "" if prediction is None else label_decode_inv[prediction]
        cv2.putText(
            img=frame,
            text=text_result,
            org=(0, 128),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=4,
            color=(255, 255, 0),
            thickness=5
        )
        cv2.imshow('Result', frame)

    stream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
