import abc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle
import xgboost as xgb
import lightgbm as lgb
import optuna
import numpy as np
from optuna.trial import Trial
import pandas as pd
from typing import List, Tuple, Union

class Model():
    def __init__(
            self, 
            xcols: List[str], 
            ycol: str,
    ) -> None:
        self._xcols = xcols
        self._ycol = ycol
        self.model = None
        self.model_name = None
        self.study = None
    
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass
    
    def get_model(self):
        return self.model
    
    def get_loss_baseline(self, y_df: pd.DataFrame) -> Tuple[float, float]:
        y_df = y_df.copy()

        if self._ycol == 'target':
            # Simplest guess: last 6 target. (Since the target needs 6 periods to reveal)
            y_df['target_shift_6'] = y_df.groupby(['stock_id', 'date_id'])[self._ycol].shift(6).fillna(0)
        elif self._ycol == 'target_change':
            # Simplest guess: change = 0
            y_df['target_shift_6'] = 0
        
        pred_loss: float = mean_absolute_error(y_df[self._ycol], y_df['target_shift_6']) # type: ignore
        base_loss: float = y_df[[self._ycol, 'target_shift_6']].corr().iloc[0, 1] # type: ignore
        return pred_loss, base_loss
    
    def evaluate(self, y_df: pd.DataFrame) -> Tuple[float, float]:
        for col in ['stock_id', 'date_id', 'seconds_in_bucket', self._ycol, 'pred']:
            assert col in y_df, f'Baseline warning: {col} not in y_df'
        y_df.sort_values(by = ['stock_id', 'date_id', 'seconds_in_bucket'], inplace = True)
        
        baseline, base_corr = self.get_loss_baseline(y_df)
        pred_loss = mean_absolute_error(y_df[self._ycol], y_df['pred'])
        pred_corr = y_df[[self._ycol, 'pred']].corr().iloc[0, 1]
        print("=" * 50)
        print("Model Name: ", self.model_name)
        print("Baseline loss: ", baseline, "Prediction loss: ", pred_loss)
        print("Baseline corr: ", base_corr, "Prediction corr: ", pred_corr)
        print("Improve:", round(1 - pred_loss / baseline, 4) * 100, "%") # type: ignore
        print("=" * 50)
        return pred_loss, pred_corr # type: ignore
    
    def train_valid_split(
            self, X: pd.DataFrame, y: pd.Series, valid_size: float = 0.2, is_sort: bool = True
    ) -> Tuple[pd.DataFrame]:
        """
            NOTE:
                X and y should be sorted by date_id.
        """
        if not is_sort:
            unique_dates = sorted(X['date_id'].drop_duplicates().values)
            split_date = unique_dates[-int(len(unique_dates) * valid_size)]
            train_X = X.query(f'date_id < {split_date}')
            valid_X = X.query(f'date_id >= {split_date}')
            train_Y = y.loc[train_X.index]
            valid_Y = y.loc[valid_X.index]

            return train_X, valid_X, train_Y, valid_Y # type: ignore
        
        else:
            head_n = int(len(X) * (1 - valid_size))
            tail_n = len(X) - head_n
            return X.head(head_n), X.tail(tail_n), y.head(head_n), y.tail(tail_n) # type: ignore

    def save_model(self, path: str) -> None:
        """path: xxx.pkl"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def visualize_tuning(self) -> None:
        """Need to install many packages. Make sure you have them."""
        try:
            fig1 = optuna.visualization.plot_optimization_history(self.study)
            fig2 = optuna.visualization.plot_slice(self.study)
            fig3 = optuna.visualization.plot_param_importances(self.study)
            for fig in [fig1, fig2, fig3]:
                fig.show()
                
        except Exception as e:
            print(e)

class LinearModel(Model):
    def __init__(self, xcols: List[str], ycol: str) -> None:
        super().__init__(xcols, ycol)
        self.model_name = 'OLS'

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        linear = LinearRegression()
        linear.fit(X, y)
        self.model = linear

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.fillna(0)
        return self.model.predict(X) # type: ignore
    
class XGBRegressModel(Model):
    def __init__(self, xcols: List[str], ycol: str) -> None:
        super().__init__(xcols, ycol)
        self.model_name = 'XGB'
        
    def tuner(self, trial: Trial):    
        params = {        
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step = 100),
            "max_depth":trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1), 
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.),
            "gamma": trial.suggest_uniform("gamma", 0.5, 1.0),
        }
        
        model = xgb.XGBRegressor(
            **params,
            objective = "reg:absoluteerror",
            device = "cuda",
            tree_method = 'gpu_hist',
            early_stopping_rounds = 100, 
        )
        
        model.fit(
            self.train_X, 
            self.train_Y, 
            eval_set = [(self.valid_X, self.valid_Y)],        
            verbose = 0
        ) 
        y_hat = model.predict(self.valid_X)
        
        return mean_absolute_error(self.valid_Y, y_hat)
    
    def train(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 10) -> None:
        self.train_X, self.valid_X, self.train_Y, self.valid_Y = self.train_valid_split(X, y) # type: ignore
        """Tuning hyper-parameters."""
        self.study = optuna.create_study(
            direction = "minimize", 
            pruner = optuna.pruners.MedianPruner(n_warmup_steps = 10)
        )
        self.study.optimize(self.tuner, n_trials = n_trials) # type: ignore
        self.best_params = self.study.best_params

        self.model = xgb.XGBRegressor(
            **self.best_params,
            tree_method = 'gpu_hist',
            device = "cuda",
            early_stopping_rounds = 50, 
        )
        self.model.fit(
            self.train_X, 
            self.train_Y, 
            eval_set=[(self.valid_X, self.valid_Y)],
            verbose = 0
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.fillna(0)
        return self.model.predict(X) # type: ignore
    
class LGBMRegressModel(Model):
    def __init__(self, xcols: List[str], ycol: str) -> None:
        super().__init__(xcols, ycol)
        self.model_name = 'LGBM'
        
    def tuner(self, trial: Trial):    
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [1000]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 1500, step = 20),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 2000, step = 100),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step = 5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step = 5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step = 0.1),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step = 0.1),
            "random_state": 42,
        }

        model = lgb.LGBMRegressor(
            **params,
            device = "cuda",
            early_stopping_rounds = 50, 
        )
        
        model.fit(
            self.train_X, 
            self.train_Y, 
            eval_set = [(self.valid_X, self.valid_Y)],        
        ) 
        y_hat = model.predict(self.valid_X)
        
        return mean_absolute_error(self.valid_Y, y_hat)
    
    def train(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 10) -> None:
        self.train_X, self.valid_X, self.train_Y, self.valid_Y = self.train_valid_split(X, y) # type: ignore
        """Tuning hyper-parameters."""
        self.study = optuna.create_study(
            direction = "minimize", 
            pruner = optuna.pruners.MedianPruner(n_warmup_steps = 10)
        )
        self.study.optimize(self.tuner, n_trials = n_trials) # type: ignore
        self.best_params = self.study.best_params

        self.model = lgb.LGBMRegressor(
            **self.best_params,
            device = "cuda",
            early_stopping_rounds = 100, 
        )
        self.model.fit(
            self.train_X, 
            self.train_Y, 
            eval_set=[(self.valid_X, self.valid_Y)],
        )



