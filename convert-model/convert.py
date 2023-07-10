import argparse
import os

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, BertConfig


def save_model(model_to_save, path_to_save="../tf-saved-model", version=1):
    export_path = os.path.join(path_to_save, str(version))
    print('export_path = {}\n'.format(export_path))
    tf.keras.models.save_model(
        model_to_save,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs and save the model in the correct way.')
    parser.add_argument('--model_path', dest='model_path', type=str, required=True,
                        help='path of the model to save (*.h5).')
    parser.add_argument('--config_path', dest='config_path', type=str,
                        help='Path to the config of the model')
    parser.add_argument('--output_path', dest='output_path', type=str,
                        help='Path to the output of the model')

    args = parser.parse_args()
    model_path = args.model_path
    config_path = args.config_path
    output_path = args.output_path

    config = BertConfig.from_pretrained(config_path)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    save_model(model, output_path)
