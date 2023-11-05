#%%
import pickle
import pandas as pd
import torch
from nn_model import FullyConnectedNet
class Predictor():
    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name

class OLSPredictor(Predictor):
    def __init__(self, path: str):
        super().__init__(path, 'OLS')

    def load_model(self):
        with open(self.path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.fillna(0)
        return self.model.predict(X)

class XGBPredictor(Predictor):
    def __init__(self, path: str):
        super().__init__(path, 'XGB')

    def load_model(self):
        with open(self.path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.fillna(0)
        return self.model.predict(X)
    
class NNPredictor(Predictor):
    def __init__(self, path: str):
        assert path.endswith('.pth'), "Invalid NN model path"
        super().__init__(path, 'NN')
    
    def load_model(self):
        self.model = FullyConnectedNet()
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.fillna(0)
        return pd.Series(self.model(
            torch.tensor(X.values, dtype=torch.float32)
        ).detach().numpy().flatten())

# %%
df = pd.read_feather('/home/sida/YIPENG/local_chatgpt/temp/US_trade/data/1104_factors_full.feather')
df_test = df.loc[df['date_id'] > 470]
# %%
nn = NNPredictor('/home/sida/YIPENG/local_chatgpt/temp/US_trade/src/USTrade/models/SimpleNN.pth')
nn.load_model()
# %%
torch.load(nn.path)
# %%
