import logging
import pandas as pd
from ludwig.api import LudwigModel
from ludwig.datasets import agnews


class AgnewsModel:

    def __init__(self, config, dataset=None, logging_level=logging.INFO):
        self.config = config
        self.model = LudwigModel(config, logging_level=logging_level)
        self.train_df, self.test_df, _ = dataset if dataset else agnews.load(split=True)
        self.train_stats = None
        self.test_stats = None
        self.predictions = None
        self.output_directory = None

    def train(self):
        self.train_stats, _, self.output_directory = self.model.train(dataset=self.train_df)

    def evaluate(self):
        self.test_stats, self.predictions, _ = self.model.evaluate(
            self.test_df, collect_predictions=True, collect_overall_stats=True
        )

    def predict(self, text_to_predict):
        self.predictions, _ = self.model.predict(text_to_predict)
        return self.predictions


if __name__ == '__main__':
    config = {
        "input_features": [
            {
                "name": "title",
                "type": "text",
                "encoder": {
                    "type": "parallel_cnn"
                }
            }
        ],
        "output_features": [
            {
                "name": "class",
                "type": "category",
            }
        ],
        "trainer": {
            "epochs": 3,
        }
    }

    model = AgnewsModel(config)

    model.train()

    model.evaluate()

    text_to_predict = pd.DataFrame({
        "title": [
            "Google may spur cloud cybersecurity M&A with $5.4B Mandiant buy",
            "Europe struggles to meet mounting needs of Ukraine's fleeing millions",
            "How the pandemic housing market spurred buyer's remorse across America",
        ]
    })

    predictions = model.predict(text_to_predict)
    print(predictions)
