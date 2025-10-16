import pandas as pd

def add_text_features(df: pd.DataFrame, text_col: str = "Product Title") -> pd.DataFrame:
    df["title_length"] = df[text_col].apply(lambda x: len(str(x)))
    df["word_count"] = df[text_col].apply(lambda x: len(str(x).split()))
    df["contains_number"] = df[text_col].apply(lambda x: any(ch.isdigit() for ch in str(x)))
    df["contains_upper"] = df[text_col].apply(lambda x: any(ch.isupper() for ch in str(x)))
    return df