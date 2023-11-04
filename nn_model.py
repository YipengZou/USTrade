import abc
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error
from typing import List, Tuple, Union

"""
    ##############################
    ##### Base NN Model Class ####
    ##############################
"""
class NNModel():
    def __init__(
            self, 
            xcols: List[str], 
            ycol: str,
    ) -> None:
        self._xcols = xcols
        self._ycol = ycol
        self.device = 'gpu'
        self.model: nn.Module
        self.train_loader: DataLoader
        self.loss: nn.Module
        self.optimizer: torch.optim.Optimizer

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass
    
    @abc.abstractmethod
    def save_model(self):
        pass

    def get_model(self):
        return self.model
    
    def get_loss_baseline(self, y_df: pd.DataFrame) -> Union[float, float]:
        y_df = y_df.copy()

        if self._ycol == 'target':
            # Simplest guess: last 6 target. (Since the target needs 6 periods to reveal)
            y_df['target_shift_6'] = y_df.groupby(['stock_id', 'date_id'])[self._ycol].shift(6).fillna(0)
        elif self._ycol == 'target_change':
            # Simplest guess: change = 0
            y_df['target_shift_6'] = 0

        return (
            mean_absolute_error(y_df[self._ycol], y_df['target_shift_6']),
            y_df[[self._ycol, 'target_shift_6']].corr().iloc[0, 1]
        ) # type: ignore
    
    def evaluate(self, y_df: pd.DataFrame) -> Tuple[float, float]:
        for col in ['stock_id', 'date_id', 'seconds_in_bucket', self._ycol, 'pred']:
            assert col in y_df, f'Baseline warning: {col} not in y_df'
        y_df.sort_values(by = ['stock_id', 'date_id', 'seconds_in_bucket'], inplace = True)
        
        baseline, base_corr = self.get_loss_baseline(y_df) # type: ignore
        pred_loss = mean_absolute_error(y_df[self._ycol], y_df['pred'])
        pred_corr = y_df[[self._ycol, 'pred']].corr().iloc[0, 1]
        print("=" * 50)
        print("Model Name: ", self.model_name) # type: ignore
        print("Baseline loss: ", baseline, "Prediction loss: ", pred_loss)
        print("Baseline corr: ", base_corr, "Prediction corr: ", pred_corr)
        print("Improve:", round(1 - pred_loss / baseline, 4) * 100, "%")
        print("=" * 50)
        return pred_loss, pred_corr # type: ignore
    
    def train_valid_split(
            self, X: pd.DataFrame, y: pd.Series, valid_size: float = 0.2, is_sort: bool = True
    ) -> [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: # type: ignore
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

            return train_X, valid_X, train_Y, valid_Y
        
        else:
            head_n = int(len(X) * (1 - valid_size))
            tail_n = len(X) - head_n
            return X.head(head_n), X.tail(tail_n), y.head(head_n), y.tail(tail_n)

    def train_fn(self):
        """Training for one epoch."""
        self.model.train()
        final_loss = 0

        for _, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x)
            loss = self.loss(y_hat.reshape(-1), y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        
        return final_loss / len(self.train_loader)

    def inference_fn(self):
        self.model.eval() # type: ignore
        preds = []

        for _, (x, y) in enumerate(self.valid_loader): # type: ignore
            x = x.to(self.device)
            with torch.no_grad():
                outputs = self.model(x) # type: ignore
            preds.append(outputs.cpu().numpy())
        preds = np.concatenate(preds).reshape(-1)

        return preds
    
class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)

        elif score < self.best_score - self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            """Want higher score"""
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

"""
    ##############################
    ####### Data Loader  #########
    ##############################
"""
    
class SimpleDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.x = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

class RNNDataset(Dataset):
    def __init__(self, factor_data: pd.DataFrame, label_data: pd.Series):
        """
            Handle the origin data.     
            Create training data for RNN / LSTM models.
            Each training point should be of the format: (periods, # of factors) -> target
            TODO:
                1. How to handle the initial data for each day?
                    Now just use yesterday data to fill, but no resonable.
                    Maybe we should just let it be 0.
                    How many 0s also represent current status in the market.
                2. We need a method to combine time info into models. Maybe develop a time factor.
        """
        factor_data = factor_data.copy()
        factor_data['target'] = label_data
        feature_cols = ut().find_pattern_cols(trained_df.columns, 'f_')
        self.data = factor_data.groupby(['stock_id']).\
            parallel_apply(lambda g: self.get_rnn_dataset_of_a_stock(
                g[feature_cols + ['time_id', 'target']], 10
            )).apply(pd.Series).stack().reset_index(drop = True)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.data[idx][0], dtype = torch.float32),
            torch.tensor(self.data[idx][1], dtype = torch.long),
        )

    def get_rnn_dataset_of_a_stock(
            self, stock_factor_df: pd.DataFrame, period: int
    ) -> List[Tuple[np.ndarray, float]]:
        """
            Input:
                stock_factor_df: pd.DataFrame
                    columns: ['f1', 'f2', ..., 'time_id', 'target']
                    pivot table of a stock
                period: int
                    how many periods to look back
        """
        record = []
        stock_factor_df = stock_factor_df.sort_values('time_id')

        for slice_df in stock_factor_df.rolling(period):
            if slice_df.shape[0] == period:  # Need a full past 10 periods
                record.append((
                    np.array(slice_df.drop(columns = ['target', 'time_id'])),  # Size: (periods, # of factors)
                    slice_df['target'].values[-1]
                ))  # Pair of factors and target, for a given stock, in a given time_id
            
            else:
                pad_rows = period - slice_df.shape[0]  # How many periods have no data. Need to fill.
                padded = np.pad(
                    np.array(slice_df.drop(columns = ['target', 'time_id'])), 
                    ((pad_rows, 0), (0, 0)), 'constant'
                )
                record.append((
                    padded, slice_df['target'].values[-1]
                ))
        return record

"""
    ##############################
    ##### Network Structure ######
    ##############################
"""

class FullyConnectedNet(nn.Module):
    def __init__(
            self, input_size: int, hidden_sizes: List[int], 
            output_size: int, dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = 'FCN'
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_prob = dropout_prob

        self.create_nn()
        
    def create_nn(self):
        # Hidden Layers
        self.hidden_layers = nn.ModuleList()
        in_features = self.input_size
        for hidden_size in self.hidden_sizes:
            layer = nn.Linear(in_features, hidden_size)
            self.hidden_layers.append(layer)
            in_features = hidden_size
        
        # Output layers
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        out = self.output_layer(x)
        return out
    
class LSTMNet(nn.Module):
    def __init__(
            self, input_dim: int, hidden_dim: int,
            num_layers: int, device: str, output_dim: int = 1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
        
"""
    ##############################
    ######## Network Model #######
    ##############################

"""
class SimpleNNModel(NNModel):
    def __init__(
            self, xcols: List[str], ycol: str,
            epoch: int, batch_size: int, learning_rate: float,
            input_size: int, hidden_sizes: List[int], output_size: int,
            dropout_prob: float
    ) -> None:
        super().__init__(xcols, ycol)
        self.model_name = 'SimpleNN'
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        torch.cuda.empty_cache()

        self.loss_list = []
        self.save_path = f'./{self.model_name}.pth'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_nn()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.loss = nn.L1Loss()
    
    def init_nn(self):
        self.model = FullyConnectedNet(
            input_size = self.input_size, 
            hidden_sizes = self.hidden_sizes, 
            output_size = self.output_size,
            dropout_prob = self.dropout_prob,
        ).to(self.device)
    
    def train(
            self, X: pd.DataFrame, y: pd.Series, 
    ) -> None:
        self.train_X, self.valid_X, self.train_Y, self.valid_Y = self.train_valid_split(X, y)
        self.train_loader = DataLoader(
            SimpleDataset(self.train_X, self.train_Y), batch_size = self.batch_size, shuffle = True
        )
        self.valid_loader = DataLoader(
            SimpleDataset(self.valid_X, self.valid_Y), batch_size = self.batch_size
        )
        es = EarlyStopping(patience = 3, mode = "min")

        for epoch in tqdm(range(self.epoch)):
            train_loss = self.train_fn()
            self.loss_list.append(train_loss)
            valid_pred = self.inference_fn()
            valid_loss = mean_absolute_error(self.valid_Y, valid_pred)
            es(valid_loss, self.model, model_path = self.save_path)
            if es.early_stop:
                print("Early stopping")
                break
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        self.init_nn()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        X = torch.tensor(X.values, dtype=torch.float32).to(self.device) # type: ignore
        with torch.no_grad():
            outputs = self.model(X)
        return pd.Series(outputs.reshape(-1).cpu().numpy())

class LSTMModel(NNModel):
    def __init__(
            self, xcols: List[str], ycol: str,
            epoch: int, batch_size: int, learning_rate: float,
            input_size: int, hidden_size: int, output_size: int,
            dropout_prob: float, num_layers: int,
    ) -> None:
        super().__init__(xcols, ycol)
        self.model_name = 'LSTM'
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        torch.cuda.empty_cache()

        self.loss_list = []
        self.save_path = f'./{self.model_name}.pth'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_nn()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.loss = nn.L1Loss()
    
    def init_nn(self):
        self.model = LSTMNet(
            input_dim = self.input_size, 
            hidden_dim = self.hidden_size, 
            num_layers = self.num_layers,
            output_dim = self.output_size,
            device = self.device, # type: ignore
        ).to(self.device)
    
    def train(
            self, X: pd.DataFrame, y: pd.Series, 
    ) -> None:
        self.RNN_data_check(X)  # Make sure 'stock_id' and 'date_id' are in X
        self.train_X, self.valid_X, self.train_Y, self.valid_Y = self.train_valid_split(X, y)
        self.train_loader = DataLoader(
            RNNDataset(self.train_X, self.train_Y), batch_size = self.batch_size, shuffle = True
        )
        self.valid_loader = DataLoader(
            RNNDataset(self.valid_X, self.valid_Y), batch_size = self.batch_size
        )
        es = EarlyStopping(patience = 3, mode = "min")

        for epoch in tqdm(range(self.epoch)):
            train_loss = self.train_fn()
            self.loss_list.append(train_loss)
            valid_pred, valid_truth = self.inference_fn()
            valid_loss = mean_absolute_error(valid_truth, valid_pred)
            es(valid_loss, self.model, model_path = self.save_path)
            if es.early_stop:
                print("Early stopping")
                break

    def RNN_data_check(self, X: pd.DataFrame) -> None:
        for col in ['stock_id', 'time_id']:
            assert col in X, f'Error from Yip: For RNN data, we need {col} in X to make dataloader'
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        self.init_nn()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        X = torch.tensor(
            X.values, dtype = torch.float32
        ).to(self.device) # type: ignore
        with torch.no_grad():
            outputs = self.model(X)
        return pd.Series(outputs.reshape(-1).cpu().numpy())
    
    def train_fn(self):
        """Training for one epoch."""
        self.model.train()
        final_loss = 0

        for _, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x)
            loss = self.loss(y_hat.reshape(-1), y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        
        return final_loss / len(self.train_loader)

    def inference_fn(self):
        self.model.eval()
        labels, preds = [], []

        for _, (x, y) in enumerate(self.valid_loader):
            x = x.to(self.device)
            with torch.no_grad():
                outputs = self.model(x)
            preds.append(outputs.cpu().numpy())
            labels.append(y)
        preds = np.concatenate(preds).reshape(-1)
        labels = np.concatenate(labels).reshape(-1)

        return preds, labels
