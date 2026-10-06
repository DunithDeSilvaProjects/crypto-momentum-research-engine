from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class Panel:
    close : pd.DataFrame
    quote_volume : pd.DataFrame
    funding: pd.DataFrame

    @property
    def ret(self) -> pd.DataFrame:
        return self.close.pct_change()

    @property
    def logret(self) -> pd.DataFrame:
        return np.log(self.close).diff()