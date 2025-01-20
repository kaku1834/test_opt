import polars as pl

def get_most_syuyaku(df):
    """Get the most common Syuyaku and related information"""
    agg_result = df.group_by("Syuyaku").agg(pl.sum("Num").alias("Num_sum"))
    max_index = agg_result["Num_sum"].arg_max()
    most_syuyaku = agg_result[max_index, "Syuyaku"]
    
    df_filtered = df.filter(pl.col('Syuyaku') == most_syuyaku)
    df_filtered_dedup = df_filtered.unique(subset=['Brand', 'Region', 'Department', 'SubCategory', 'Syuyaku'])
    
    most_syuyaku_Brand = df_filtered_dedup['Brand'].to_list()[0]
    most_syuyaku_Region = df_filtered_dedup['Region'].to_list()[0]
    most_syuyaku_Department = df_filtered_dedup['Department'].to_list()[0]
    most_syuyaku_SubCategory = df_filtered_dedup['SubCategory'].to_list()[0]

    return most_syuyaku_Brand, most_syuyaku_Region, most_syuyaku_Department, most_syuyaku_SubCategory, most_syuyaku

def get_sorted_unique_values(_df, column):
    """Get sorted unique values from a column"""
    return sorted(_df[column].unique())

def filter_data_sequentially(df, selected_Brand=None, 
                            selected_Region=None, 
                            selected_Department=None, 
                            selected_SubCategory=None, 
                            selected_Syuyaku=None, 
                            selected_Size=None, 
                            selected_Color=None, 
                            selected_SKU=None, 
                            start_date=None, 
                            end_date=None):
    """Filter data based on multiple criteria"""
    # Single selection filters
    if selected_Brand:
        df = df.filter(pl.col('Brand') == selected_Brand)
    if selected_Region:
        df = df.filter(pl.col('Region') == selected_Region)
    if selected_Department:
        df = df.filter(pl.col('Department') == selected_Department)
    if selected_SubCategory:
        df = df.filter(pl.col('SubCategory') == selected_SubCategory)
    if selected_Syuyaku:
        df = df.filter(pl.col('Syuyaku') == selected_Syuyaku)
    
    # Multiple selection filters
    if selected_Size:
        df = df.filter(pl.col('Size').is_in(selected_Size))
    if selected_Color:
        df = df.filter(pl.col('Color').is_in(selected_Color))
    if selected_SKU:
        df = df.filter(pl.col('SKU').is_in(selected_SKU))

    # Date range filter
    if start_date and end_date:
        df = df.filter((pl.col('Date') >= pl.lit(start_date)) & (pl.col('Date') <= pl.lit(end_date)))
    
    return df