def get_summary_stats(df):
    summary = df.groupby('category').mean(numeric_only=True).reset_index()
    return summary