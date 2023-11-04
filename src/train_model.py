#%%
import pandas as pd
import polars as pl
import os
os.chdir('/home/sida/YIPENG/local_chatgpt/temp/US_trade/src/USTrade')
from model import Model, LinearModel, XGBRegressModel
from nn_model import SimpleNNModel
from FactorGenerator import ut
df = pd.read_feather('/home/sida/YIPENG/local_chatgpt/temp/US_trade/data/1104_factors_full.feather')
df = df.loc[df['date_id'] >= 460]
df_train = df.loc[df['date_id'] <= 470]
df_test = df.loc[df['date_id'] > 470]
# %%
def train_ols(train_df: pd.DataFrame, test_df: pd.DataFrame):
    test_df = test_df.copy()
    ols_model = LinearModel(
        xcols = sorted(ut().find_pattern_cols(list(train_df.columns), 'f_')),
        ycol = 'target',
    )
    ols_model.train(train_df[ols_model._xcols], train_df[ols_model._ycol])
    test_df['pred'] = ols_model.predict(test_df[ols_model._xcols])
    ols_model.evaluate(test_df)
    return ols_model

def train_xgb(train_df: pd.DataFrame, test_df: pd.DataFrame, n_trials: int = 20):
    xgb_model = XGBRegressModel(
        xcols = sorted(ut().find_pattern_cols(list(train_df.columns), 'f_')),
        ycol = 'target',
    )
    xgb_model.train(
        train_df[xgb_model._xcols], train_df[xgb_model._ycol], n_trials
    )
    test_df['pred'] = xgb_model.predict(test_df[xgb_model._xcols])
    xgb_model.evaluate(test_df)
    xgb_model.visualize_tuning()
    return xgb_model

def train_simple_nn(train_df: pd.DataFrame, test_df: pd.DataFrame, num_epoch: int = 100):
    test_df = test_df.copy()
    nn_model = SimpleNNModel(
        xcols = sorted(ut().find_pattern_cols(list(train_df.columns), 'f_')),
        ycol = 'target',
        epoch = num_epoch,
        batch_size = 1024,
        learning_rate = 0.001,
        input_size = len((ut().find_pattern_cols(list(train_df.columns), 'f_'))),
        hidden_sizes = [128, 64],
        output_size = 1,
        dropout_prob = 0.1,
    )
    nn_model.train(train_df[nn_model._xcols], train_df[nn_model._ycol])
    test_df['pred'] = nn_model.predict(test_df[nn_model._xcols]).values
    nn_model.evaluate(test_df)
    return nn_model

# ols_model = train_ols(df_train, df_test)
# xgb_model = train_xgb(df_train, df_test, 50)
nn_model = train_simple_nn(df_train, df_test, 10)
# %%
