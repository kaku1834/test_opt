import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib
import numpy as np
import time
from functools import lru_cache

@lru_cache(maxsize=1)
def setup_plot_style():
    """Setup common plot styles and colors with caching"""
    return {
        'sales': '#B02A29',
        'prediction': '#00A1E4',
        'purple': 'purple',
        'orange': 'orange',
        'black': 'black',
        'stock_colors': ['#E4C087', '#F3F3E0'],
        'size_colors': ['#133E87', '#CBDCEB', '#BC7C7C', '#F6EFBD']
    }

def create_dashboard_figure(df_disp, df_real, df_pred, color_cols, size_cols, select_date='2023-10-03'):
    """Create the main dashboard figure with all subplots"""
    colors = setup_plot_style()
    plot_timings = {}
    
    # 共通の設定を一度に行う
    plt.rcParams.update({
        'figure.dpi': 300,
        'figure.figsize': (22, 36),
        'font.size': 8,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold'
    })
    
    # サブプロットの作成を最適化
    fig, axes = plt.subplots(nrows=13, ncols=1, 
                            gridspec_kw={'hspace': 0.6, 
                                       'height_ratios': [0.5, 0.5, 0.5, 0.2, 1.5, 0.2, 1.2, 0.2, 1.2, 0.1, 1.5, 0.5, 1.5]})
    
    # 共通の軸設定を一括適用
    for ax in axes:
        if not ax.get_visible():  # スペーシング用の非表示軸はスキップ
            continue
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # プロット関数とその引数のマッピング
    plot_functions = [
        (plot_limited_results, [axes[0], df_disp, colors]),
        (plot_holidays, [axes[1], df_disp, colors]),
        (plot_temperature, [axes[2], df_disp, colors]),
        (plot_sales_prediction, [axes[4], df_disp, df_real, df_pred, colors]),
        (plot_stock_by_color, [axes[6], df_disp, color_cols, colors]),
        (plot_stock_by_size, [axes[8], df_disp, size_cols, colors]),
        (plot_out_of_stock, [axes[10], df_disp, colors]),
        (plot_customers, [axes[11], df_disp, colors]),
        (plot_selling_stores, [axes[12], df_disp, colors])
    ]
    
    # 各プロット関数を実行し、タイミングを記録
    for func, args in plot_functions:
        start = time.time()
        func(*args)
        plot_timings[func.__name__] = time.time() - start
    
    # スペーシング用の軸を非表示に
    for i in [3, 5, 7, 9]:
        axes[i].axis('off')
    
    # x軸ラベルの重複を防ぐ
    for ax in [axes[0], axes[1], axes[10], axes[11]]:
        ax.label_outer()
    
    return fig, plot_timings

def plot_limited_results(ax, df, colors):
    """Plot limited results subplot"""
    ax.set_title("限定実績", fontsize=16, fontweight='bold', color='black')
    ax.bar(df['Date'], df['Rate'], color=colors['sales'], label='Limited Results', alpha=0.5, linewidth=2)
    ax.set_ylabel('Limited Results')
    ax.set_yticks([0, 3, 4, 7])
    ax.tick_params(axis='y')
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), borderaxespad=0, frameon=False)

def plot_holidays(ax, df_disp, colors):
    """Plot holidays subplot"""
    df_disp_events = df_disp.dropna(subset=['EventNum'])
    max_event_num = int(df_disp_events['EventNum'].max() if not df_disp_events.empty else 0)
    
    ax.set_title("祝日数", fontsize=16, fontweight='bold', color='black')
    ax.bar(df_disp_events['Date'], df_disp_events['EventNum'], color=colors['sales'], label='Holidays', width=1.5)
    ax.set_ylabel('Holidays')
    ax.set_yticks(list(range(0, max_event_num + 1)))
    ax.tick_params(axis='y')
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), borderaxespad=0, frameon=False)
    ax.grid(True)
    ax.tick_params(axis='x', rotation=90, labelsize=8)

def plot_temperature(ax, df_disp, colors):
    """Plot temperature subplot"""
    ax.set_title("気温", fontsize=16, fontweight='bold', color='black')
    ax.plot(df_disp['Date'], df_disp['Temperature'], color=colors['black'], label='Temperature', alpha=0.5)
    ax.set_ylabel('Temperature')
    ax.tick_params(axis='y')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=90, labelsize=8)

def plot_sales_prediction(ax, df_disp, df_real, df_pred, colors):
    """Plot sales and prediction subplot"""
    ax.set_title("売上実績と予測値", fontsize=16, fontweight='bold', color='black')
    ax.bar(df_disp['Date'], df_disp['Sales'], color=colors['prediction'], label='Sales', alpha=0.5)
    ax.bar(df_disp[df_disp['Date'] > '2023-10-01']['Date'], 
           df_disp[df_disp['Date'] > '2023-10-01']['Sales'], 
           color='orange', label='Sales2', alpha=1)
    ax.plot(df_disp['Date'], df_disp['SalesPred'], color=colors['purple'], 
            label='SalesPred', alpha=0.5, linewidth=2)
    ax.set_ylabel('Sales')
    ax.tick_params(axis='y')
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), borderaxespad=0, frameon=False)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=90, labelsize=8)

def plot_stock_by_color(ax, df_disp, color_cols, colors):
    """Plot stock by color subplot with optimized performance"""
    dates = df_disp['Date'].values  # NumPy配列に変換して再利用
    bottom = np.zeros(len(dates))
    
    for i, color in enumerate(color_cols):
        values = df_disp[color].values  # NumPy配列に変換
        ax.bar(dates, values, bottom=bottom,
               color=colors['stock_colors'][i], 
               edgecolor=colors['stock_colors'][i], 
               width=0.5, label=color, alpha=0.5)
        bottom += values
    
    ax.set_title("在庫数(色別)")
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), borderaxespad=0, frameon=False)
    ax.set_ylabel('Color')

def plot_stock_by_size(ax, df_disp, size_cols, colors):
    """Plot stock by size subplot"""
    bar_width = 0.5
    bottom_values = np.zeros(len(df_disp['Date']))
    
    for i, size in enumerate(size_cols):
        ax.bar(df_disp['Date'], df_disp[size], bottom=bottom_values,
               color=colors['size_colors'][i], edgecolor=colors['size_colors'][i], 
               width=bar_width, label=size, alpha=0.5)
        bottom_values += df_disp[size]
    
    ax.set_title("在庫数(サイズ別)", fontsize=16, fontweight='bold', color='black')
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), borderaxespad=0, frameon=False)
    ax.set_ylabel('Size')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=90, labelsize=8)

def plot_out_of_stock(ax, df_disp, colors):
    """Plot out of stock stores subplot"""
    ax2 = ax.twinx()
    
    ax.set_title("欠品店舗数", fontsize=16, fontweight='bold', color='black')
    ax.plot(df_disp['Date'], df_disp['OutOfStockStores'], color='grey', 
            label='OutOfStockStores', alpha=1, linewidth=2)
    ax.set_ylabel('OutOfStockStores')
    ax.tick_params(axis='y', direction='in')
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), borderaxespad=0, frameon=False)

    ax2.plot(df_disp['Date'], df_disp['OutOfStockRate'], color='red', 
             label='OutOfStockRate', linestyle='dashed', alpha=0.8, linewidth=2)
    ax2.set_ylabel('OutOfStockRate')
    ax2.tick_params(axis='y', direction='in')
    ax2.yaxis.set_label_coords(1.005, 0.5)
    for label in ax2.get_yticklabels():
        label.set_x(0.97)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.005, 0.9), borderaxespad=0, frameon=False)

def plot_customers(ax, df_disp, colors):
    """Plot customers subplot"""
    ax.set_title("来客数", fontsize=16, fontweight='bold', color='black')
    ax.plot(df_disp['Date'], df_disp['Customers'], color=colors['orange'], 
            label='Customers', alpha=1, linewidth=2)
    ax.set_ylabel('Customers')
    ax.tick_params(axis='y')
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), borderaxespad=0, frameon=False)

def plot_selling_stores(ax, df_disp, colors):
    """Plot selling stores subplot"""
    ax3 = ax.twinx()
    
    ax.set_title("販売店舗数", fontsize=16, fontweight='bold', color='black')
    ax.bar(df_disp['Date'], df_disp['SellingStores'], color='#B98E68', 
           label='SellingStores', width=1.5, alpha=0.5)
    ax.set_ylabel('SellingStores')
    ax.tick_params(axis='y', direction='in')
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1), borderaxespad=0, frameon=False)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=90, labelsize=8)

    ax3.plot(df_disp['Date'], df_disp['TotalStores'], color='#C62368', 
             label='TotalStores', alpha=0.5, linewidth=2)
    ax3.set_ylabel('TotalStores')
    ax3.tick_params(axis='y', direction='in')
    ax3.yaxis.set_label_coords(1.005, 0.5)
    for label in ax3.get_yticklabels():
        label.set_x(0.97)
    ax3.legend(loc='upper left', bbox_to_anchor=(1.005, 0.9), borderaxespad=0, frameon=False)

# Add other plotting functions for each component... 