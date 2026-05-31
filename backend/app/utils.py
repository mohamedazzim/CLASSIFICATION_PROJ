from typing import Any, Dict
import pandas as pd


def prepare_input(payload: Dict[str, Any]):
    # Convert input dict to DataFrame
    df = pd.DataFrame([payload])
    return df
