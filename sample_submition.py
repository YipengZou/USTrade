%%time
import pandas as pd
import numpy as np
from typing import List, Tuple, Union
import polars as pl
pl.enable_string_cache(enable = True)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import abc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from pandarallel import pandarallel
# pandarallel.initialize(progress_bar = False)
pandarallel.initialize(progress_bar = True, nb_workers = 4)
class ut():
    """常用的函数放到一个类里面"""
    def __init__(self):
        pass
    
    def find_pattern_cols(self, cols: List[str], pattern: str, direction: int = 1) -> List[str]:
        if direction == 1:
            return [col for col in cols if col.startswith(pattern)]
        if direction == -1:
            return [col for col in cols if col.endswith(pattern)]

    def handle_abnormal_val(self, df: Union[pl.DataFrame, pd.DataFrame], col: str, val: float = 0) -> pd.DataFrame:
        if isinstance(df, pl.DataFrame):
            return df.with_columns(
                pl.when(
                    pl.col(col).is_infinite() |
                    pl.col(col).is_nan() |
                    pl.col(col).is_null()
                )
                .then(val)
                .otherwise(pl.col(col))
                .keep_name()
            ).to_pandas()
        
        elif isinstance(df, pd.DataFrame):
            df[col] = df[col].replace(-np.inf, val)
            df[col] = df[col].replace(np.inf, val)
            df[col] = df[col].replace(np.nan, val)

            return df
"""
    //TODO: 
        1. 试一下log_transform能加到哪里
        2. 用polars dataframe 代替pandas dataframe

"""

class FactorGenerator():
    def __init__(
            self, 
            df: pd.DataFrame, 
            mode: str
    ) -> None:
        """
            mode: str
                train / test
        """
        self.ut = ut()
        self._df = df
        self._df = self._df.sort_values(by = ['stock_id', 'date_id', 'seconds_in_bucket'])  # Defensive movement.
        self._mode = mode

    """Basic Data Generator"""
    def gen_basis_spread_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """spread因子, ask_price和bid_price的差值"""
        df['spread'] = df.eval('ask_price - bid_price')
        return self.ut.handle_abnormal_val(df, 'spread', 0)
    
    def gen_basis_mid_price_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """mid_price因子, ask_price和bid_price的中间值"""
        df['mid_price'] = df.eval('ask_price + bid_price') / 2
        
        return self.ut.handle_abnormal_val(df, 'mid_price', 1)
    
    """Self created factors"""
    def gen_target_change_g(self, g: pd.DataFrame) -> pd.DataFrame:
        """得到target的变化额度而不是绝对值. 预测变化额度然后叠加上一s的真实target"""
        g['target_change'] = self.ut.handle_abnormal_val(g['target'].diff(), 0)

        return self.ut.handle_abnormal_val(g, 'target_change', 0)

    def gen_price_mwap_g(self, g: pd.DataFrame) -> pd.DataFrame:
        """使用reference price和在当前reference新增的match数量来计算mwap (match-weighted average price)"""
        match_diff = g['matched_size'].diff().fillna(0)
        match_cum = match_diff.cumsum()
        mwap = (match_diff * g['reference_price']).cumsum() / match_cum  # match的均价
        g.loc[:, 'mwap'] = self.ut.handle_abnormal_val(mwap, 1)

        return self.ut.handle_abnormal_val(g, 'mwap', 1)

    def gen_factor_imb_chg_g(self, g: pd.DataFrame) -> pd.DataFrame:
        """得到imbalance变化比例相对于match变化比例的幅度"""
        g['f_imb_chg'] = g['imbalance_size'].diff() / g['matched_size'].diff()

        return self.ut.handle_abnormal_val(g, 'f_imb_chg', 0)

    def gen_factor_bid_size_increase_g(self, g: pd.DataFrame, p: int = 10) -> pd.DataFrame:
        """判断某个时刻bid_size是否激增,与过去一段时间bid_size的数量比较"""
        g[f'f_bid_size_incr_ewm_{p}'] = g['bid_size'] / g['bid_size'].ewm(span = p).mean()  # 10个period的指数平均
        g = self.ut.handle_abnormal_val(g, f'f_bid_size_incr_ewm_{p}', 1)
        g['f_bid_size_incr_expand'] = g['bid_size'] / g['bid_size'].expanding().mean()  # 当天开始到该时刻的平均
        return self.ut.handle_abnormal_val(g, f'f_bid_size_incr_expand', 1)

    def gen_factor_ask_size_increase_g(self, g: pd.DataFrame, p: int = 10) -> pd.DataFrame:
        """判断某个时刻ask_size是否激增,与过去一段时间ask_size的数量比较"""
        g[f'ask_size_incr_ewm_{p}'] = g['ask_size'] / g['ask_size'].ewm(span = p).mean()  # 10个period的指数平均
        g = self.ut.handle_abnormal_val(g, f'ask_size_incr_ewm_{p}', 1)
        g['f_ask_size_incr_expand'] = g['ask_size'] / g['ask_size'].expanding().mean()  # 当天开始到该时刻的平均

        return self.ut.handle_abnormal_val(g, 'f_ask_size_incr_expand', 1)

    def gen_factor_momentum_g(self, g: pd.DataFrame, n: int = 6, price: str = 'mwap') -> pd.DataFrame:
        """动量因子"""
        g[f'f_momentum_{price}_{n}'] = (g[price] / g[price].shift(n) - 1) * 100

        return self.ut.handle_abnormal_val(g, f'f_momentum_{price}_{n}', 0)

    """Idea from papers"""
    """Group by factors"""
    def gen_factor_volume_order_imbalance_g(self, g: pd.DataFrame) -> pd.DataFrame:
        """
            订单失衡因子. Source: https://bigquant.com/wiki/doc/20200709-gYLf5LSvmq
            logic of bid_size_change:
                1. bid_price up -> new bid_size
                2. bid_price same -> bid_size diff
                3. bid_price down -> 0
            logic of ask_size_change:
                1. ask_price up -> 0
                2. ask_price same -> ask_size diff
                3. ask_price down -> new ask_size
                
            Output:
                f_voi: 原始版本的订单失衡因子, size差值的绝对值. 但绝对值在不同股票中没有可比性
                f_voi_demean: 考虑了当天开盘以来所有f_voi的均值然后除掉. 绝对值变成相对值. 计算均值的时候只考虑非0的term
        """
        bid_up = g['bid_price'] > g['bid_price'].shift(1)
        bid_same = g['bid_price'] == g['bid_price'].shift(1)
        bid_size_change = g['bid_size'].diff() * bid_same + g['bid_size'] * bid_up
        
        ask_down = g['ask_price'] > g['ask_price'].shift(1)
        ask_same = g['ask_price'] == g['ask_price'].shift(1)
        ask_size_change = g['ask_size'].diff() * ask_same + g['ask_size'] * ask_down

        g['f_voi'] = bid_size_change - ask_size_change
        g = self.ut.handle_abnormal_val(g, 'f_voi', 0)

        g['f_voi_demean'] = g['f_voi'] / g['f_voi'].abs().replace(0, np.nan).expanding().mean()  # 计算过往均值的时候只考虑非0的
        return self.ut.handle_abnormal_val(g, 'f_voi_demean', 0)
    
    def gen_factor_order_imbalance_ratio_g(self, g: pd.DataFrame) -> pd.DataFrame:
        """
            订单失衡率因子. Source: https://bigquant.com/wiki/doc/20200709-gYLf5LSvmq
            (bv1 - av1) / (bv1 + av1)
        """
        g['f_oir'] = g.eval('bid_size - ask_size') / g.eval('bid_size + ask_size')

        return self.ut.handle_abnormal_val(g, 'f_oir', 0)
    
    def gen_factor_spread_momentum_g(self, g: pd.DataFrame, p: int = 6) -> pd.DataFrame:
        """
            spread动量因子, 当spread增加时意味着某一方强烈进攻, 一口气吃掉了对手盘几档的价位
            动量: 当前spread - 过去p个period的spread均值
        """
        # rolling, 不包含最后一个
        g[f'f_spread_momentum_{p}'] = g['spread'] - \
            g['spread'].rolling(p, closed = 'left', min_periods = 1).mean()
        
        return self.ut.handle_abnormal_val(g, f'f_spread_momentum_{p}', 0)

    def gen_factor_mid_price_momentum_g(self, g: pd.DataFrame, p: int = 6) -> pd.DataFrame:
        """mid_price动量因子"""
        g[f'f_mid_price_momentum_{p}'] = g['mid_price'] / g['mid_price'].shift(p) - 1

        return self.ut.handle_abnormal_val(g, f'f_mid_price_momentum_{p}', 0)
    
    def gen_factor_spread_momentum_direction_g(
            self, g: pd.DataFrame, p_spread: int = 6, p_mid: int = 6
    ) -> pd.DataFrame:
        """
            结合mid_price和spread的动量因子
            判断spread的变化是由于bid_price的变化还是ask_price的变化
        """
        sign = (g[f'f_mid_price_momentum_{p_mid}'] > 0).astype(float) * 2 - 1  # 1: up, -1: down
        g[f'f_spread_momentum_direction_{p_spread}_{p_mid}'] = g[f'f_spread_momentum_{p_spread}'] * sign

        return self.ut.handle_abnormal_val(g, f'f_spread_momentum_direction_{p_spread}_{p_mid}', 0)

    """DataFrame Factors"""
    def gen_factor_midprice_vwap_diff_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            mid_price和vwap的差值, 看当前偏向买方还是卖方
                diff > 0: ask1 > bid1, 偏向卖方
                diff < 0: ask1 < bid1, 偏向买方
        """
        df['f_midprice_vwap_diff'] = df.eval('mid_price - wap')

        return self.ut.handle_abnormal_val(df, 'f_midprice_vwap_diff', 0)
    
    """Generate all factors"""
    def gen_factor_g(self, g: pd.DataFrame) -> pd.DataFrame:
        """所有需要groupby后计算的因子放在这"""
        if self._mode != 'test':
            g = self.gen_target_change_g(g)
        g = self.gen_price_mwap_g(g)

        g = self.gen_factor_imb_chg_g(g)
        g = self.gen_factor_bid_size_increase_g(g)
        g = self.gen_factor_ask_size_increase_g(g)
        g = self.gen_factor_momentum_g(g, 6, 'mwap')
        g = self.gen_factor_volume_order_imbalance_g(g)
        g = self.gen_factor_order_imbalance_ratio_g(g)
        g = self.gen_factor_spread_momentum_g(g, 6)
        g = self.gen_factor_mid_price_momentum_g(g, 6)
        g = self.gen_factor_spread_momentum_direction_g(g, 6, 6)
        return g
    
    def gen_factor_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """所有对dataframe直接操作的因子放在这"""
        df = self.gen_basis_spread_df(df)
        df = self.gen_basis_mid_price_df(df)
        df = self.gen_factor_midprice_vwap_diff_df(df)
        return df
    
    def section_normalization(self, g: pd.DataFrame) -> pd.DataFrame:
        """
            对每一列的异常值都进行处理了
            根据date_id和seconds_in_bucket的截面来做normalization
        """
        factors = self.ut.find_pattern_cols(g.columns, 'f_')
        for fac in factors:
            g = self.ut.handle_abnormal_val(g, fac, 0)  # Handle abnormal values first
            g[f'{fac}_norm'] = (g[fac] - g[fac].mean()) / g[fac].std()
            g = self.ut.handle_abnormal_val(g, f'{fac}_norm', 0)

        return g

    def run(self) -> pd.DataFrame:
        res: pd.DataFrame = self._df.copy()
        res: pd.DataFrame = self.gen_factor_df(res)
        res: pd.DataFrame = (
            res.groupby(['stock_id', 'date_id'])
            .parallel_apply(lambda g: self.gen_factor_g(g))
        ).reset_index(drop = True)
        res: pd.DataFrame = (
            res.groupby(['date_id', 'seconds_in_bucket'])
            .parallel_apply(lambda g: self.section_normalization(g))
        ).reset_index(drop = True)  # Also handled all the abnormal values. No need to handle further.
        res = res.sort_values(['stock_id', 'date_id', 'seconds_in_bucket'])
        return res
    
    def run2(self) -> pd.DataFrame:
        res: pd.DataFrame = self._df.clone()
        res: pd.DataFrame = self.gen_factor_df(res)

"""MAIN"""
# df = pl.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
# df = df.filter(~df['target'].is_null())
# df_use = df.filter(pl.col('date_id') >= 470).to_pandas()

# gen = FactorGenerator(df_use, 'train')
# trained_df = gen.run()
# trained_df['time_id'] = trained_df['date_id'] * 1000 + trained_df['seconds_in_bucket']