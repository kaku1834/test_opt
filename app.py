##-0 Library導入-##
import streamlit as st # UI画面
import streamlit_shadcn_ui as ui
import numpy as np # データ処理
import polars as pl
import pandas as pd
import time
from collections import defaultdict
from modules.data_loader import load_data
from modules.data_processor import get_most_syuyaku, get_sorted_unique_values, filter_data_sequentially
from modules.visualizer import create_dashboard_figure
from modules.data_transformer import prepare_visualization_data
from modules.auth_utils import setup_page_config, initialize_authentication
from functools import lru_cache

# Setup page configuration
setup_page_config()

# Initialize authentication
authenticator, auth_status = initialize_authentication()

# Initialize timing dictionary
timing_stats = defaultdict(float)

def measure_time(step_name):
    """Decorator to measure execution time of functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            timing_stats[step_name] = end_time - start_time
            return result
        return wrapper
    return decorator

# キャッシュ付きのデータ読み込み関数
@st.cache_data
def cached_load_data():
    return load_data()

# キャッシュ付きの集計関数
@st.cache_data
def cached_get_most_syuyaku(_raw):
    return get_most_syuyaku(_raw)

# キャッシュ付きのユニーク値取得
@st.cache_data
def cached_get_sorted_unique_values(_data, column):
    return get_sorted_unique_values(_data, column)

# さらなる最適化として、データフレームの変換処理もキャッシュ
@st.cache_data
def cached_prepare_visualization_data(_raw_filtered, _dateInfo, _rate, selected_Region, selected_Syuyaku):
    return prepare_visualization_data(_raw_filtered, _dateInfo, _rate, selected_Region, selected_Syuyaku)

# Main application logic
if auth_status:
    # キャッシュを使用したデータ読み込み
    raw, dateInfo, rate, metricTable, itemInfoTable, skuSummary, AccuracySummary, ProblemSummary = cached_load_data()
    
    # キャッシュを使用した初期処理
    most_syuyaku_Brand, most_syuyaku_Region, most_syuyaku_Department, most_syuyaku_SubCategory, most_syuyaku = cached_get_most_syuyaku(raw)
    
    # フィルタリングでキャッシュを使用
    sorted_Brand = cached_get_sorted_unique_values(raw, 'Brand')
    selected_Brand = st.sidebar.selectbox("グローバル事業", sorted_Brand, index = sorted_Brand.index(most_syuyaku_Brand), key="brand_select")
    raw_filtered = filter_data_sequentially(raw, selected_Brand=selected_Brand)

    # フィルタ2: 事業を選択した上で、国を選択
    sorted_Region = cached_get_sorted_unique_values(raw_filtered, 'Region')
    selected_Region = st.sidebar.selectbox("事業国", sorted_Region, index = sorted_Region.index(most_syuyaku_Region), key="region_select")
    raw_filtered = filter_data_sequentially(raw_filtered, selected_Region=selected_Region)

    # フィルタ3: 事業と国選択した上で、部門を選択
    sorted_Department = cached_get_sorted_unique_values(raw_filtered, 'Department')
    selected_Department = st.sidebar.selectbox("部門", sorted_Department, index = sorted_Department.index(most_syuyaku_Department), key="dept_select")
    raw_filtered = filter_data_sequentially(raw_filtered, selected_Department=selected_Department)

    # フィルタ4: 事業、国、部門を選択した上で、サブカテゴリを選択
    sorted_SubCategory = cached_get_sorted_unique_values(raw_filtered, 'SubCategory')
    selected_SubCategory = st.sidebar.selectbox("サブカテゴリ", sorted_SubCategory, index = sorted_SubCategory.index(most_syuyaku_SubCategory), key="subcat_select")
    raw_filtered = filter_data_sequentially(raw_filtered, selected_SubCategory=selected_SubCategory)

    # フィルタ5: すべての以前の選択に基づいた集約の選択
    sorted_Syuyaku = cached_get_sorted_unique_values(raw_filtered, 'Syuyaku')
    selected_Syuyaku = st.sidebar.selectbox("販売集約", sorted_Syuyaku, key="syuyaku_select")
    raw_filtered = filter_data_sequentially(raw_filtered, selected_Syuyaku=selected_Syuyaku)

    # フィルタ6: 選択された集約のうち、サイズを選択
    sorted_Size = cached_get_sorted_unique_values(raw_filtered, 'Size')
    selected_Size = st.sidebar.multiselect("サイズ", sorted_Size, placeholder = "未入力の場合、全選択される", key="size_select")
    raw_filtered = filter_data_sequentially(raw_filtered, selected_Size=selected_Size)

    # フィルタ7: 選択された集約のうち、色を選択
    sorted_Color = cached_get_sorted_unique_values(raw_filtered, 'Color')
    selected_Color = st.sidebar.multiselect("カラー", sorted_Color, placeholder = "未入力の場合、全選択される", key="color_select")    
    raw_filtered = filter_data_sequentially(raw_filtered, selected_Color=selected_Color)

    # フィルタ8: 選択された集約のうち、SKUを選択
    sorted_SKU = cached_get_sorted_unique_values(raw_filtered, 'SKU')
    selected_SKU = st.sidebar.multiselect("SKU", sorted_SKU, placeholder = "未入力の場合、全選択される", key="sku_select")
    raw_filtered = filter_data_sequentially(raw_filtered, selected_SKU=selected_SKU)

    # 日付範囲の選択
    start_date = st.sidebar.date_input("開始日", raw.select(pl.col("Date")).min().to_series()[0])
    end_date = st.sidebar.date_input("終了日", raw.select(pl.col("Date")).max().to_series()[0])

    # ユーザ操作によって、フィルタされたデータ
    raw_filtered = filter_data_sequentially(raw_filtered, start_date=start_date, end_date=end_date)

    # Get unique product information for the selected Syuyaku
    raw_unique = raw.unique(subset=['Department', 'Syuyaku', 'SKU', 'Color', 'Size', 'Length']).select(['Department', 'Syuyaku', 'Tanpin', 'Color', 'Size', 'Length'])
    ProductInfo = raw_unique.filter(pl.col('Syuyaku') == selected_Syuyaku)
    ProductInfoDF = ProductInfo.to_pandas()
    ProductInfoDF.columns = ['部門', '販売集約', '単品', 'カラー',  'サイズ',  'レングス']

    # Filter summary tables for selected Syuyaku
    AccuracySummaryDF = AccuracySummary.query('Syuyaku == @selected_Syuyaku').iloc[:, 1:]
    ProblemSummaryDF = ProblemSummary.query('Syuyaku == @selected_Syuyaku').iloc[:, 1:]

    st.title('商品属性情報')
    ui.table(ProductInfoDF)

    st.title("精度サマリ")
    ui.table(AccuracySummaryDF)       
    
    st.title("既存課題検知結果サマリ")
    ui.table(ProblemSummaryDF)
    
    st.title("課題検知のための情報の可視化")

    # データ変換とプロット生成を最適化
    df_disp, df_real, df_pred, transform_timings = cached_prepare_visualization_data(
        raw_filtered, dateInfo, rate, selected_Region, selected_Syuyaku
    )

    # Measure data transformation time
    start_time = time.time()
    total_transform_time = time.time() - start_time
    timing_stats['データ変換処理 (全体)'] = total_transform_time
    timing_stats['データ変換処理 (その他)'] = total_transform_time - sum(transform_timings.values())

    # Add transformation timings to main timing stats
    for step_name, step_time in transform_timings.items():
        timing_stats[f'データ変換: {step_name}'] = step_time

    # Measure column preparation time
    start_time = time.time()
    size_cols = raw_filtered.select(pl.col("Size")).unique().to_series().to_list()
    color_cols = raw_filtered.select(pl.col("Color")).unique().to_series().to_list()
    timing_stats['カラム準備'] = time.time() - start_time

    # Measure visualization time
    start_time = time.time()
    with st.spinner('グラフを生成中...'):
        fig, plot_timings = create_dashboard_figure(df_disp, df_real, df_pred, color_cols, size_cols)
        st.pyplot(fig)
    total_viz_time = time.time() - start_time
    timing_stats['グラフ生成 (全体)'] = total_viz_time
    timing_stats['グラフ生成 (プロット以外)'] = total_viz_time - sum(plot_timings.values())

    # Add individual plot timings to timing_stats
    for plot_name, plot_time in plot_timings.items():
        timing_stats[f'プロット: {plot_name}'] = plot_time

    # Calculate total processing time first
    total_time = (
        timing_stats['データ読み込み'] +
        timing_stats['初期データ処理'] +
        timing_stats['データ変換処理 (全体)'] +  # または詳細時間の合計
        timing_stats['カラム準備'] +
        timing_stats['グラフ生成 (全体)']  # または詳細時間の合計
    )

    # Create and display timing summary
    timing_df = pd.DataFrame({
        '処理ステップ': timing_stats.keys(),
        '所要時間 (秒)': [f"{t:.2f}" for t in timing_stats.values()],
        '割合 (%)': [f"{(t/total_time)*100:.1f}" for t in timing_stats.values()],
        'カテゴリ': ['データ処理' if not k.startswith(('プロット', 'グラフ')) 
                   else ('グラフ描画' if k.startswith('プロット') else 'グラフ全体')
                   for k in timing_stats.keys()]
    })
    
    # Define the order of processing steps
    step_order = [
        # データ処理カテゴリ
        'データ読み込み',
        '初期データ処理',
        'データ変換処理 (全体)',
        'データ変換処理 (その他)',
        'データ変換: 在庫データ処理',
        'データ変換: 売上データ処理',
        'データ変換: レート処理',
        'データ変換: データ結合',
        'カラム準備',
        # グラフ全体カテゴリ
        'グラフ生成 (全体)',
        'グラフ生成 (プロット以外)',
        # グラフ描画カテゴリ
        'プロット: 限定実績プロット',
        'プロット: 祝日プロット',
        'プロット: 気温プロット',
        'プロット: 売上予測プロット',
        'プロット: 色別在庫プロット',
        'プロット: サイズ別在庫プロット',
        'プロット: 欠品プロット',
        'プロット: 来客数プロット',
        'プロット: 販売店舗プロット'
    ]
    
    # Create a categorical type with our custom order
    timing_df['処理ステップ'] = pd.Categorical(
        timing_df['処理ステップ'], 
        categories=step_order, 
        ordered=True
    )
    
    # Sort by our custom order
    timing_df = timing_df.sort_values('処理ステップ')

    # Display timing summary
    st.title("処理時間分析")
    ui.table(timing_df)
    
    st.caption(f"総処理時間: {total_time:.2f} 秒")
