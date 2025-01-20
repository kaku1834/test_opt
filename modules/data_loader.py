import polars as pl
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    """Load and cache all data files"""
    raw = pl.read_csv('./data/raw_add01.csv')
    dateInfo = pl.read_csv('./data/dateInfo_v2.csv')
    rate = pl.read_csv('./data/rate.csv')
    metricTable = pd.read_csv("./data/metric_table.csv")
    itemInfoTable = pd.read_csv("./data/itemInfo_table.csv")      
    skuSummary = pd.read_csv("./data/SKUSummary.csv") 
    AccuracySummary = pd.read_csv("./data/AccuracySummary.csv") 
    ProblemSummary = pd.read_csv("./data/ProblemSummary.csv") 

    # Convert Date columns to datetime
    raw = raw.with_columns(pl.col('Date').str.strptime(pl.Datetime))
    dateInfo = dateInfo.with_columns(pl.col('Date').str.strptime(pl.Datetime))
    rate = rate.with_columns(pl.col('Date').str.strptime(pl.Datetime))

    return raw, dateInfo, rate, metricTable, itemInfoTable, skuSummary, AccuracySummary, ProblemSummary 