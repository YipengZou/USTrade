from typing import List, Tuple, Union
import polars as pl
import pandas as pd

class FactorEvaluator():
    """
        评估某个因子的有效性
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        xcols: List[str], 
        ycols: List[str],
    ) -> None:
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        self._df: pl.DataFrame = df.fill_nan(0)  # Force fill.
        self._xcols = xcols
        self._ycols = ycols
        self.common_cols = ['stock_id', 'date_id', 'seconds_in_bucket']
        self.check()

    def check(self):
        for col in self.common_cols:
            if col not in self.common_cols + self._xcols + self._ycols:
                raise ValueError(f'Column {col} not in df')
    
    def get_y_stats(self, ycol: str):
        """得到每个股票某个y值的分布, 用于z值转化真是值"""
        y_stat = pd.concat([self._df.groupby(['stock_id'])[ycol].mean(), 
                           self._df.groupby(['stock_id'])[ycol].std()], axis = 1)
        y_stat.columns = ['mean', 'std']
        return y_stat
    
    def get_daily_pcor(self, xcol: str, ycol: str, method: str = 'pearson') -> Union[pd.Series, float]:
        """得到每一日factor & target的pcor"""
        res = (
            self._df.group_by(['stock_id', 'date_id'])
            .agg(pl.corr(xcol, ycol, method = method))
            .fill_nan(0)  # deal with xcol keeps the same while ycol changes
            .group_by('date_id')
            .mean()
            .sort('date_id')
        )
        return res, res[col].mean()
    
    def get_seconds_group_return(self, g: pl.DataFrame, xcol: str, ycol: str, n_cut: int = 5) -> pl.Series:
        """g: 某个seconds所有的股票及return"""
        try:
            g = (
                g.with_columns(
                    g[xcol]
                    .qcut(n_cut, labels = ['T'+str(i) for i in range(1, n_cut+1)])
                    .alias('label')
                )  # Assign lables
            )
            g_ret = (
                g.group_by('label')
                .agg(pl.mean(ycol))
            )

        except Exception as e:
            g_ret = pl.DataFrame({
                'label': ['T'+str(i) for i in range(1, n_cut+1)], 
                'target': [0.0] * n_cut
            })

        g_ret = (
            g_ret.with_columns(
                g_ret['label'].cast(pl.Categorical).alias('label'),
                g['date_id'].head(1).alias('date_id'),
                g['seconds_in_bucket'].head(1).alias('seconds_in_bucket'),
            )
        )
        return g_ret

    def get_group_return(self, xcol: str, ycol: str, n_cut: int = 5) -> pd.DataFrame:
        """
            得到股票根据某个因子排序后每日各组的收益. 
            polars用来加速计算
            返回pd.DataFrame方便后续使用
        """
        g_ret = (
            ev._df
            .group_by(['date_id', 'seconds_in_bucket'])
            .map_groups(
                lambda g: self.get_seconds_group_return(g, xcol, ycol, n_cut)
            ).sort(['date_id', 'seconds_in_bucket'])
        )

        g_ret_pd =(
            g_ret.to_pandas()
            .pivot(index = ['date_id', 'seconds_in_bucket'], columns = 'label', values = 'target')
        )
        g_ret_pd = g_ret_pd[['T'+str(i) for i in range(1, n_cut+1)]]
        g_ret_pd['long-short'] = g_ret_pd['T1'] - g_ret_pd['T5']

        return g_ret_pd

    def polt_group_return(self, xcol: str, ycol: str, n_cut: int = 5, pcor: float = 0):
        g_ret_pd = self.get_group_return(xcol, ycol, n_cut)
        pcor = round(pcor, 4) * 100
        g_ret_pd.cumsum().plot(
            title = f'group return of {xcol}\nWith IC: {pcor}%', 
            figsize=(10, 5)
        )

    def check_mae(self, xcol: str, ycol: str):
        """
            1. 根据date_id + seconds进行groupby, 得到每个股票在截面的rank
            2. 把rank转换成percentile, 转换成z-score
            3. 根据每个stock_id过往的target分布的mean, std来将每个tick的z-score转换成predict作为预测
            4. 计算predict & true的MAE, baseline为predict全部为0
        """
        def get_z_score(g: pd.DataFrame, xcol: str) -> pd.Series:
            """根据rank转换成z-score"""
            percentile = (g[xcol].rank() - 0.5) / len(g)
            return pd.Series(stats.norm.ppf(percentile), index = percentile.index)
        
        df_use = self._df[self.common_cols + [xcol, ycol]].copy()
        df_use['z'] = e._df.groupby(['date_id', 'seconds_in_bucket']).\
                        apply(lambda g: get_z_score(g, xcol)).\
                        reset_index(level = [0,1], drop = True).sort_index()

        y_stat = self.get_y_stats(ycol)
        df_use = pd.merge(df_use, y_stat, on = 'stock_id')
        df_use['predict'] = df_use['z'] * df_use['std'] + df_use['mean']
        df_use['baseline'] = 0
        
        baseline = df_use[['target_change', 'baseline']].diff(axis=1).iloc[:, 1].abs().mean()
        prediction = df_use[['target_change', 'predict']].diff(axis=1).iloc[:, 1].abs().mean()
        improve = round(1 - prediction / baseline, 4) * 100
        print(f"MAE of baseline: {round(baseline, 4)}")
        print(f"MAE of prediction: {round(prediction, 4)}")
        print(f"Improvement of factor {xcol} over naive baseline is: {improve}%")
        return df_use
        
ev = FactorEvaluator(
    df = temp,
    xcols = sorted(ut().find_pattern_cols(temp.columns, 'f_')),
    ycols = ['target_change']
)
for col in ev._xcols[:]:
    print(f"Factor: {col}")
    print(ev.get_daily_pcor(col, 'target', 'spearman')[1])
    print(ev.get_daily_pcor(col, 'target', 'pearson')[1])
    # ev.polt_group_return(col, 'target', 5, ev.get_daily_pcor(col, 'target')[1])