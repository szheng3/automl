import torch
import pandas as pd
from ludwig.models.inference import InferenceModule
import json

from ludwig.utils.inference_utils import to_inference_module_input_from_dataframe

if __name__ == '__main__':


    text_to_predict = pd.DataFrame({
        "title": [
            "Google may spur cloud cybersecurity M&A with $5.4B Mandiant buy",
            "Europe struggles to meet mounting needs of Ukraine's fleeing millions",
            "How the pandemic housing market spurred buyer's remorse across America",
        ]
    })

    inference_module = InferenceModule.from_directory('./results/api_experiment_run/model/')
    # output_df = inference_module.predict(text_to_predict)
    scripted_module = torch.jit.script(inference_module)

    with open( f"results/api_experiment_run/model/model_hyperparameters.json") as f:
        config = json.load(f)

    input_sample_dict = to_inference_module_input_from_dataframe(text_to_predict, config)
    # output_df = scripted_module.predict(text_to_predict)

    print(scripted_module(input_sample_dict))



