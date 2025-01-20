import polars as pl
import pandas as pd
import time

def process_stock_data(raw_filtered):
    """Process stock data for visualization"""
    # コラム選定
    stock = raw_filtered.select(['Date', 'Size', 'Color', 'Stock'])
    
    # サイズの週合計
    stock_size = (stock.drop('Color')
                 .group_by(['Size', pl.col('Date').dt.truncate('1w') + pl.duration(days = 6)])
                 .agg(pl.col('Stock').sum())
                 .with_columns(pl.col('Stock').cast(pl.Int64))
                 .sort(['Size', 'Date']))
    
    # 色の週合計
    stock_color = (stock.drop('Size')
                  .group_by(['Color', pl.col('Date').dt.truncate('1w') + pl.duration(days = 6)])
                  .agg(pl.col('Stock').sum())
                  .with_columns(pl.col('Stock').cast(pl.Int64))
                  .sort(['Color', 'Date']))
    
    # サイズ、色週合計の縦結合
    stock_size = stock_size.rename({'Size': 'Var'})
    stock_color = stock_color.rename({'Color': 'Var'})
    stock = pl.concat([stock_size, stock_color], how='vertical')
    
    # 週ごとに列名を[S, M, L, Color1, Color2] に転置させる
    stock_pivoted = stock.pivot(index='Date', 
                              on='Var', 
                              values='Stock', 
                              aggregate_function='first')
    
    return stock_pivoted

def process_sales_data(raw_filtered, dateInfo, selected_Region):
    """Process sales data for visualization"""
    dateInfo_byCountry = dateInfo.filter(pl.col('cntry_cd') == selected_Region)

    # 日付が重複しているため、売上を日別に合計処理
    raw_salesD = (raw_filtered.group_by('Date')
                 .agg(pl.col('Sales').sum(), 
                      pl.col('SalesPred').sum())
                 .sort('Date'))

    # 日別売上と日付情報を結合
    raw_salesD_dateInfo = dateInfo_byCountry.join(raw_salesD, on='Date', how='right')

    # 週合計が不要な列を分離
    raw_salesD_dateInfo_1 = raw_salesD_dateInfo.select(['Date', 'Temperature', 'Event'])
    
    # 週合計が必要な列を分離
    raw_salesD_dateInfo_2 = raw_salesD_dateInfo.select(
        ['Date', 'Sales', 'SalesPred', 'Customers', 'EventNum']
    )
    
    # 週合計
    raw_salesD_dateInfo_2ed = (raw_salesD_dateInfo_2.group_by_dynamic('Date', 
                            every='1w',
                            start_by='monday',
                            closed='right',
                            label='right',
                            offset='6d')
                            .agg(pl.all().sum()))

    # 週合計が不要な列と週合計が必要な列を結合、null値の補填
    raw_salesW_dateInfo = (raw_salesD_dateInfo_1
                          .join(raw_salesD_dateInfo_2ed, on='Date', how='left')
                          .with_columns((pl.col('Event').is_not_null())
                          .cast(pl.Int8).alias('Holiday')))
    
    return raw_salesW_dateInfo

def process_rate_data(rate, selected_Syuyaku):
    """Process rate data for visualization"""
    return rate.filter(pl.col('Syuyaku') == selected_Syuyaku)

def combine_all_data(raw_salesW_dateInfo, stock_pivoted, rate_filtered):
    """Combine all processed data and handle missing values"""
    # 週合計売上と週合計在庫の結合
    raw_salesW_dateInfo_stockW = raw_salesW_dateInfo.join(stock_pivoted, on='Date', how='left')

    # null値の補填
    required_columns = ["Date", "L", "M", "S", "Color1", "Color2"]
    
    for col in required_columns:
        if col not in stock_pivoted.columns:
            stock_pivoted = stock_pivoted.with_columns(pl.lit(0).cast(pl.Int64).alias(col)) 
    stock_pivoted = stock_pivoted.select(required_columns)

    # 週別の売上、在庫、日付情報付き(来客数、気温、イベント数)
    df_disp_pl = raw_salesW_dateInfo_stockW.join(rate_filtered, on='Date', how='left')

    # null値の補填処理
    columns_to_fill = [
        "Sales", "SalesPred", "Customers", "Rate",
        "L", "M", "S", "Color1", "Color2", 
        "OutOfStockStores", "OutOfStockRate",
        "TotalStores", "SellingStores"
    ]

    df_disp_pl = df_disp_pl.with_columns([
        pl.col(col).fill_null(strategy="backward") 
        for col in columns_to_fill 
        if col in df_disp_pl.columns
    ])

    return df_disp_pl

def prepare_visualization_data(raw_filtered, dateInfo, rate, selected_Region, selected_Syuyaku):
    """Main function to prepare all data for visualization"""
    timing_stats = {}
    
    start = time.time()
    stock_pivoted = process_stock_data(raw_filtered)
    timing_stats['在庫データ処理'] = time.time() - start
    
    start = time.time()
    raw_salesW_dateInfo = process_sales_data(raw_filtered, dateInfo, selected_Region)
    timing_stats['売上データ処理'] = time.time() - start
    
    start = time.time()
    rate_filtered = process_rate_data(rate, selected_Syuyaku)
    timing_stats['レート処理'] = time.time() - start
    
    start = time.time()
    df_disp_pl = combine_all_data(raw_salesW_dateInfo, stock_pivoted, rate_filtered)
    timing_stats['データ結合'] = time.time() - start
    
    # Convert to pandas for visualization
    df_disp = df_disp_pl.to_pandas()
    df_disp['Date'] = pd.to_datetime(df_disp['Date'])

    # Split data for before/after comparison
    select_date = '2023-10-03'
    df_real = df_disp[df_disp['Date'] <= select_date]
    df_pred = df_disp[df_disp['Date'] > select_date]
    
    return df_disp, df_real, df_pred, timing_stats 