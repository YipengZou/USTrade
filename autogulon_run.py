#%%
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common import space
from autogluon.core.dataset import TabularDataset
from autogluon.tabular.predictor.predictor import TabularPredictor

#%%
def fit_specific_model(
        df: TabularDataset, model_name: str, dir: str, time_limit: int = 20, 
) -> TabularPredictor:
    assert model_name in ['GBM', 'CAT', 'XGB', 'RF', 'XT', 'KNN', 'LR', 'NN_TORCH', 'FASTAI']
    num_gpus = 2 if model_name in ['XGB', 'NN_TORCH'] else 0
    save_path = dir + model_name
    hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
        'num_trials': 5, # try at most 5 different hyperparameter configurations for each type of model
        'scheduler' : 'local',
        'searcher': 'auto', # to tune hyperparameters using random search routine with a local scheduler
    }  # Refer to TabularPredictor.fit docstring for all valid values

    predictor = TabularPredictor(
         label = 'target', eval_metric = 'mean_absolute_error', path = save_path,
    ).fit(
        df,
        presets = 'best_quality',
        time_limit = time_limit,
        hyperparameters = {model_name: {}},
        hyperparameter_tune_kwargs = hyperparameter_tune_kwargs,
        num_gpus = num_gpus, # type: ignore
    )
    return predictor

if __name__ == '__main__':
    df_use = pd.read_feather('/home/sida/YIPENG/local_chatgpt/temp/US_trade/data/factors_10.feather')
    train_data = TabularDataset(df_use)
    save_dir = '/home/sida/YIPENG/AutogluonModels/_trained/'
    for model_type in ['GBM', 'CAT', 'XGB', 'RF', 'XT', 'KNN', 'LR', 'NN_TORCH', 'FASTAI']:
        predictor = fit_specific_model(train_data, model_type, save_dir, time_limit = 120)
        print(predictor.leaderboard())
