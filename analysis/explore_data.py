"""
PUMS Data Exploration and Analysis

This module provides functions for exploring and visualizing ACS PUMS data
to understand variable distributions and relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
import warnings

# Import codebook for meaningful labels
from src.codebook import (
    PERSON_VARIABLE_LABELS,
    HOUSEHOLD_VARIABLE_LABELS,
    SEX_VALUES,
    RAC1P_VALUES,
    SCHL_VALUES,
    ESR_VALUES,
    COW_VALUES,
    TEN_VALUES,
    HHT_VALUES,
    BLD_VALUES,
    HUPAC_VALUES,
    STATE_CODES,
    get_variable_label,
    get_value_label,
    get_education_category,
    get_state_name,
)

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory for reports
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def load_data(filepath: str, sample_frac: Optional[float] = None) -> pd.DataFrame:
    """Load PUMS CSV data.
    
    Args:
        filepath: Path to CSV file
        sample_frac: Optional fraction to sample (for large files)
    
    Returns:
        DataFrame with PUMS data
    """
    df = pd.read_csv(filepath, low_memory=False)
    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    return df


def generate_summary_report(df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    """Generate a summary report of all variables.
    
    Args:
        df: DataFrame to summarize
        name: Name for the report
    
    Returns:
        Summary DataFrame
    """
    summary = []
    
    for col in df.columns:
        col_data = df[col]
        info = {
            'variable': col,
            'dtype': str(col_data.dtype),
            'non_null': col_data.notna().sum(),
            'null_count': col_data.isna().sum(),
            'null_pct': round(col_data.isna().mean() * 100, 2),
            'unique': col_data.nunique(),
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            info['min'] = col_data.min()
            info['max'] = col_data.max()
            info['mean'] = round(col_data.mean(), 2)
            info['median'] = col_data.median()
            info['std'] = round(col_data.std(), 2)
        else:
            info['min'] = None
            info['max'] = None
            info['mean'] = None
            info['median'] = None
            info['std'] = None
            
        summary.append(info)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(REPORT_DIR / f"{name}_summary.csv", index=False)
    print(f"Summary report saved to {REPORT_DIR / f'{name}_summary.csv'}")
    return summary_df


def plot_distributions(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (16, 12)) -> None:
    """Plot distributions of numeric variables.
    
    Args:
        df: DataFrame
        numeric_cols: List of columns to plot (auto-detected if None)
        figsize: Figure size
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude ID columns and weights
        numeric_cols = [c for c in numeric_cols if c not in ['serialno', 'year', 'sporder']]
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i, col in enumerate(numeric_cols):
        if i >= len(axes):
            break
        ax = axes[i]
        data = df[col].dropna()
        
        # Use histogram for continuous, bar for categorical-like
        if data.nunique() <= 20:
            data.value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue')
        else:
            ax.hist(data, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        
        ax.set_title(f'{col}', fontsize=10)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribution plots saved to {REPORT_DIR / 'distributions.png'}")


def plot_correlation_matrix(df: pd.DataFrame, cols: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (14, 12)) -> pd.DataFrame:
    """Plot correlation matrix for numeric variables.
    
    Args:
        df: DataFrame
        cols: Columns to include (auto-detected if None)
        figsize: Figure size
    
    Returns:
        Correlation matrix DataFrame
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if c not in ['serialno', 'year', 'sporder']]
    
    corr = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 8})
    
    ax.set_title('Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    corr.to_csv(REPORT_DIR / 'correlation_matrix.csv')
    print(f"Correlation matrix saved to {REPORT_DIR / 'correlation_matrix.png'}")
    return corr


def plot_income_analysis(df: pd.DataFrame) -> None:
    """Analyze income-related variables.
    
    Args:
        df: DataFrame with income columns
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Income distribution (log scale)
    if 'hincp' in df.columns:
        ax = axes[0, 0]
        income = df['hincp'].dropna()
        income_positive = income[income > 0]
        ax.hist(np.log10(income_positive), bins=50, color='steelblue', edgecolor='white')
        ax.set_title('Household Income Distribution (log10)')
        ax.set_xlabel('log10(Income)')
        ax.set_ylabel('Count')
    
    # Income by state - using codebook labels
    if 'hincp' in df.columns and 'st' in df.columns:
        ax = axes[0, 1]
        state_income = df.groupby('st')['hincp'].median().sort_values(ascending=False).head(15)
        state_income.index = state_income.index.map(lambda x: get_state_name(str(x).zfill(2)))
        state_income.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Median Household Income by State (Top 15)')
        ax.set_xlabel('Median Income')
    
    # Income vs household size
    if 'hincp' in df.columns and 'np' in df.columns:
        ax = axes[1, 0]
        size_income = df.groupby('np')['hincp'].median()
        size_income = size_income[size_income.index <= 10]  # Limit to 10 persons
        size_income.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Median Income by Household Size')
        ax.set_xlabel('Number of Persons')
        ax.set_ylabel('Median Income')
    
    # Income by tenure - using codebook labels
    if 'hincp' in df.columns and 'ten' in df.columns:
        ax = axes[1, 1]
        tenure_income = df.groupby('ten')['hincp'].median()
        tenure_income.index = tenure_income.index.map(lambda x: TEN_VALUES.get(x, f'Code {x}'))
        tenure_income.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'Median Income by {get_variable_label("ten", "household")}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'income_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Income analysis saved to {REPORT_DIR / 'income_analysis.png'}")


def plot_geographic_analysis(df: pd.DataFrame) -> None:
    """Analyze geographic distribution.
    
    Args:
        df: DataFrame with geographic columns
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Records by state - using codebook labels
    if 'st' in df.columns:
        ax = axes[0]
        state_counts = df['st'].value_counts().head(20)
        state_counts.index = state_counts.index.map(lambda x: get_state_name(str(x).zfill(2)))
        state_counts.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Records by State (Top 20)')
        ax.set_xlabel('Number of Records')
    
    # Weighted population by state - using codebook labels
    if 'st' in df.columns and 'wgtp' in df.columns:
        ax = axes[1]
        state_pop = df.groupby('st')['wgtp'].sum().sort_values(ascending=False).head(20)
        state_pop.index = state_pop.index.map(lambda x: get_state_name(str(x).zfill(2)))
        state_pop = state_pop / 1e6  # Convert to millions
        state_pop.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Estimated Households by State (Millions, Top 20)')
        ax.set_xlabel('Households (Millions)')
    
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'geographic_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Geographic analysis saved to {REPORT_DIR / 'geographic_analysis.png'}")


def plot_housing_analysis(df: pd.DataFrame) -> None:
    """Analyze housing characteristics.
    
    Args:
        df: DataFrame with housing columns
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Building type - using codebook labels
    if 'bld' in df.columns:
        ax = axes[0, 0]
        bld_counts = df['bld'].value_counts().sort_index()
        bld_counts.index = bld_counts.index.map(lambda x: BLD_VALUES.get(x, f'{x}'))
        bld_counts.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(get_variable_label('bld', 'household'))
    
    # Tenure - using codebook labels
    if 'ten' in df.columns:
        ax = axes[0, 1]
        tenure_counts = df['ten'].value_counts().sort_index()
        tenure_counts.index = tenure_counts.index.map(lambda x: TEN_VALUES.get(x, f'{x}'))
        tenure_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=sns.color_palette("husl", len(tenure_counts)))
        ax.set_title(get_variable_label('ten', 'household'))
        ax.set_ylabel('')
    
    # Household size
    if 'np' in df.columns:
        ax = axes[1, 0]
        size_counts = df['np'].value_counts().sort_index()
        size_counts = size_counts[size_counts.index <= 10]
        size_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Household Size Distribution')
        ax.set_xlabel('Number of Persons')
    
    # Household type - using codebook labels
    if 'hht' in df.columns:
        ax = axes[1, 1]
        hht_counts = df['hht'].value_counts().sort_index()
        hht_counts.index = hht_counts.index.map(lambda x: HHT_VALUES.get(x, f'{x}'))
        hht_counts.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(get_variable_label('hht', 'household'))
    
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'housing_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Housing analysis saved to {REPORT_DIR / 'housing_analysis.png'}")


def plot_cost_burden_analysis(df: pd.DataFrame) -> None:
    """Analyze housing cost burden.
    
    Args:
        df: DataFrame with cost columns
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rent burden (GRPIP)
    if 'grpip' in df.columns:
        ax = axes[0]
        grpip = df['grpip'].dropna()
        grpip = grpip[(grpip > 0) & (grpip <= 100)]
        
        # Create burden categories
        bins = [0, 30, 50, 100]
        labels = ['<30% (Affordable)', '30-50% (Burdened)', '>50% (Severely Burdened)']
        burden = pd.cut(grpip, bins=bins, labels=labels)
        burden_counts = burden.value_counts()
        
        colors = ['green', 'orange', 'red']
        burden_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors)
        ax.set_title('Renter Cost Burden')
        ax.set_ylabel('')
    
    # Owner burden (OCPIP)
    if 'ocpip' in df.columns:
        ax = axes[1]
        ocpip = df['ocpip'].dropna()
        ocpip = ocpip[(ocpip > 0) & (ocpip <= 100)]
        
        bins = [0, 30, 50, 100]
        labels = ['<30% (Affordable)', '30-50% (Burdened)', '>50% (Severely Burdened)']
        burden = pd.cut(ocpip, bins=bins, labels=labels)
        burden_counts = burden.value_counts()
        
        colors = ['green', 'orange', 'red']
        burden_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors)
        ax.set_title('Owner Cost Burden')
        ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'cost_burden_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Cost burden analysis saved to {REPORT_DIR / 'cost_burden_analysis.png'}")


def generate_full_report(filepath: str, sample_frac: float = 0.1) -> None:
    """Generate a complete analysis report.
    
    Args:
        filepath: Path to PUMS CSV file
        sample_frac: Fraction to sample for faster analysis
    """
    print(f"\n{'='*60}")
    print("PUMS Data Analysis Report")
    print(f"{'='*60}\n")
    
    print(f"Loading data from {filepath}...")
    df = load_data(filepath, sample_frac=sample_frac)
    print(f"Loaded {len(df):,} records ({sample_frac*100:.0f}% sample)\n")
    
    print("Generating summary statistics...")
    summary = generate_summary_report(df, "household")
    print(f"\nDataset has {len(df.columns)} variables:\n")
    print(summary[['variable', 'dtype', 'non_null', 'null_pct', 'unique']].to_string(index=False))
    
    print("\n" + "-"*60)
    print("Generating visualizations...")
    
    plot_distributions(df)
    plot_correlation_matrix(df)
    plot_income_analysis(df)
    plot_geographic_analysis(df)
    plot_housing_analysis(df)
    plot_cost_burden_analysis(df)
    
    print(f"\n{'='*60}")
    print(f"Report complete! All outputs saved to {REPORT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/pums_household_2023.csv"
    sample = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    
    generate_full_report(filepath, sample_frac=sample)
