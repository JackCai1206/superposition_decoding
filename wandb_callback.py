from transformers.integrations import WandbCallback
import pandas as pd
from transformers import EvalPrediction, Trainer

# def decode_predictions(tokenizer, pred: EvalPrediction, n_streams=1):
#     ground_truth = []
#     prediction = []
#     for s in range(n_streams):
#         labels = tokenizer.batch_decode(pred.label_ids[s], skip_special_tokens=True)
#         prediction_text = tokenizer.batch_decode(pred.predictions[0][s].argmax(axis=-1), skip_special_tokens=True)
#         ground_truth.append(labels)
#         prediction.append(prediction_text)
#     return [{"ground_truth": ground_truth[i], "prediction": prediction[i]} for i in range(n_streams)]

def decode_predictions(tokenizer, prompt_ids, pred_ids, n_streams=1):
    prompt = []
    prediction = []
    for s in range(n_streams):
        pred_text = tokenizer.batch_decode(pred_ids[s], skip_special_tokens=True)
        prediction.append(pred_text)
        prompt_text = tokenizer.batch_decode(prompt_ids[s], skip_special_tokens=True)
        prompt.append(prompt_text)
    return [{"Prompt": prompt[i], "Generation": prediction[i]} for i in range(n_streams)]

class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each 
    logging step during training. It allows to visualize the 
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset 
          for generating predictions.
        num_samples (int, optional): Number of samples to select from 
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer: Trainer, tokenizer, val_dataset,
                 num_samples=100, freq=2, args=None):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated 
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from 
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq
        self.my_args = args
      
    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        self._wandb.config.update(self.my_args, allow_val_change=True)

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
        if int(state.epoch) % self.freq == 0:
            inputs = self.sample_dataset[0:1]
            prompt_ids = inputs["input_ids"].permute(1,0,2)[..., :40].unbind(0)
            pred_ids = self.trainer.model.generate(prompt_ids, max_new_tokens=100, use_past_kv_cache=False)
            data = decode_predictions(self.tokenizer, prompt_ids, pred_ids, self.my_args.n_streams)
            results = {}
            for i, stream_pred in enumerate(data):
              predictions_df = pd.DataFrame(stream_pred)
              predictions_df["epoch"] = state.epoch
              records_table = self._wandb.Table(dataframe=predictions_df)
              results[f"pred_stream_{i}"] = records_table
            self._wandb.log(results)
            
            # ----------------- (Legacy) -----------------
            # # generate predictions
            # predictions = self.trainer.predict(self.sample_dataset)
            # # decode predictions and labels
            # predictions = decode_predictions(self.tokenizer, predictions, self.my_args.n_streams)
            # # add predictions to a wandb.Table
            # results = {}
            # for i, stream_pred in enumerate(predictions):
            #   predictions_df = pd.DataFrame(stream_pred)
            #   predictions_df["epoch"] = state.epoch
            #   records_table = self._wandb.Table(dataframe=predictions_df)
            #   results[f"pred_stream_{i}"] = records_table
            # # log the table to wandb
            # self._wandb.log(results)

