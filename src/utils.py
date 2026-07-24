import pandas as pd

def get_trading_date(dt_series: pd.Series) -> pd.Series:
    """
    根据给定的时间序列计算交易日。
    每日早晨 06:00 作为交易日的分界线。
    """
    shifted = dt_series - pd.Timedelta(hours=6)
    if hasattr(shifted, 'dt'):
        return shifted.dt.date.values
    else:
        return shifted.date
