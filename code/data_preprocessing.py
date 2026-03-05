"""
data_preprocessing.py
=====================
Loads, cleans, and merges the five UK Biodiversity Indicator datasets into a
single analysis-ready dataframe.

Datasets
--------
1. Pollinating Insects (target variable) – mean occupancy index, 1980-2024
2. Butterflies / Insects of Wider Countryside – abundance indices, 1976-2024
3. Plants of Wider Countryside – habitat-specific plant indices, 2015-2024
4. Agri-Environment Schemes – land area in higher/entry-level schemes, 1992-2022
5. Habitat Connectivity – butterfly functional connectivity, 1985-2012
"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _project_root():
    """Return the project root directory (parent of code/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _data_dir():
    """Return the path to the Downloads folder containing raw data."""
    return os.path.expanduser("~/Downloads")


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def load_pollinating_insects(filepath=None):
    """
    Load the pollinating insects indicator (target variable).

    Returns a DataFrame with columns: Year, Pollinator_Index
    Index is set so 1980 = 100.
    """
    if filepath is None:
        filepath = os.path.join(_data_dir(), "UK-BDI-2025-pollinating-insects.xlsx")

    df = pd.read_excel(filepath, sheet_name='1', engine='openpyxl', header=None)

    # Find the header row containing 'Year'
    header_idx = None
    for i in range(min(10, len(df))):
        row_vals = df.iloc[i].astype(str).tolist()
        if 'Year' in row_vals:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header row in pollinating insects dataset")

    # Extract data below the header
    data = df.iloc[header_idx + 1:, :2].copy()
    data.columns = ['Year', 'Pollinator_Index']
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    data['Pollinator_Index'] = pd.to_numeric(data['Pollinator_Index'], errors='coerce')
    data = data.dropna(subset=['Year'])
    data['Year'] = data['Year'].astype(int)
    data = data.reset_index(drop=True)

    print(f"  Pollinating Insects: {len(data)} rows, years {data['Year'].min()}-{data['Year'].max()}")
    return data


def load_butterflies(filepath=None):
    """
    Load butterfly abundance indices from the Insects of Wider Countryside dataset.

    Extracts three smoothed indices from Tables 1-3:
      - Butterfly_All_Species (all 50 species)
      - Butterfly_Specialists (habitat specialist species)
      - Butterfly_Generalists (generalist species)

    Returns a DataFrame indexed by Year.
    """
    if filepath is None:
        filepath = os.path.join(_data_dir(), "UK-BDI-2025-insects-wider-countryside.xlsx")

    df = pd.read_excel(filepath, sheet_name='1', engine='openpyxl', header=None)

    # Find the sub-header row (contains 'Year', 'Unsmoothed index', 'Smoothed index')
    header_idx = None
    for i in range(min(15, len(df))):
        row_vals = df.iloc[i].astype(str).tolist()
        if 'Year' in row_vals and 'Smoothed index' in row_vals:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header row in butterflies dataset")

    # The sheet has 7 tables side by side separated by blank columns.
    # Table 1: cols 0-4 (all-species), Table 2: cols 6-10 (specialists),
    # Table 3: cols 12-16 (generalists)
    data_start = header_idx + 1

    # Table 1: All species – col 0 = Year, col 2 = Smoothed index
    t1 = df.iloc[data_start:, [0, 2]].copy()
    t1.columns = ['Year', 'Butterfly_All_Species']

    # Table 2: Specialists – col 6 = Year, col 8 = Smoothed index
    t2 = df.iloc[data_start:, [6, 8]].copy()
    t2.columns = ['Year', 'Butterfly_Specialists']

    # Table 3: Generalists – col 12 = Year, col 14 = Smoothed index
    t3 = df.iloc[data_start:, [12, 14]].copy()
    t3.columns = ['Year', 'Butterfly_Generalists']

    # Convert types and merge
    result = None
    for tbl in [t1, t2, t3]:
        col_name = tbl.columns[1]
        tbl = tbl.copy()
        tbl.iloc[:, 0] = pd.to_numeric(tbl.iloc[:, 0], errors='coerce')
        tbl.iloc[:, 1] = pd.to_numeric(tbl.iloc[:, 1], errors='coerce')
        tbl = tbl.dropna(subset=[tbl.columns[0]])
        tbl['Year'] = tbl.iloc[:, 0].astype(int)
        tbl = tbl[['Year', col_name]].reset_index(drop=True)

        if result is None:
            result = tbl
        else:
            result = result.merge(tbl, on='Year', how='outer')

    result = result.sort_values('Year').reset_index(drop=True)
    print(f"  Butterflies: {len(result)} rows, years {result['Year'].min()}-{result['Year'].max()}")
    return result


def load_plants(filepath=None):
    """
    Load plant abundance indices for UK broad habitat types.

    Pivots the long-format data so each habitat becomes a column:
      - Plants_Arable
      - Plants_Bog_Wet_Heath
      - Plants_Lowland_Grassland
      - Plants_Woodland

    Returns a DataFrame indexed by Year.
    """
    if filepath is None:
        filepath = os.path.join(_data_dir(), "UK-BDI-2025-plants-wider-countryside.xlsx")

    df = pd.read_excel(filepath, sheet_name='1', engine='calamine', header=None)

    # Find header row
    header_idx = None
    for i in range(min(10, len(df))):
        row_vals = df.iloc[i].astype(str).tolist()
        if 'Habitat' in row_vals and 'Year' in row_vals:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header row in plants dataset")

    data = df.iloc[header_idx + 1:, :3].copy()
    data.columns = ['Habitat', 'Year', 'Unsmoothed_Index']
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    data['Unsmoothed_Index'] = pd.to_numeric(data['Unsmoothed_Index'], errors='coerce')
    data = data.dropna(subset=['Year', 'Habitat'])
    data['Year'] = data['Year'].astype(int)

    # Create clean habitat names for column pivoting
    habitat_map = {
        'Arable': 'Plants_Arable',
        'Bog & wet heath': 'Plants_Bog_Wet_Heath',
        'Lowland grassland': 'Plants_Lowland_Grassland',
        'Broadleaved woodland & hedges': 'Plants_Woodland',
    }
    data['Habitat_Clean'] = data['Habitat'].map(habitat_map)
    data = data.dropna(subset=['Habitat_Clean'])

    # Pivot to wide format
    result = data.pivot_table(
        index='Year', columns='Habitat_Clean', values='Unsmoothed_Index'
    ).reset_index()
    result.columns.name = None

    print(f"  Plants: {len(result)} rows, years {result['Year'].min()}-{result['Year'].max()}")
    return result


def load_agri_environment(filepath=None):
    """
    Load agri-environment scheme data (higher-level and entry-level).

    Aggregates area across all UK countries by year, producing:
      - Agri_Env_Higher_Area  (million hectares, summed across UK)
      - Agri_Env_Entry_Area   (million hectares, summed across UK)

    Returns a DataFrame indexed by Year.
    """
    if filepath is None:
        filepath = os.path.join(_data_dir(), "UK-BDI-2025-agri-environment-schemes.xlsx")

    results = {}
    sheet_names = {'1': 'Agri_Env_Higher_Area', '2': 'Agri_Env_Entry_Area'}

    for sheet, col_name in sheet_names.items():
        df = pd.read_excel(filepath, sheet_name=sheet, engine='openpyxl', header=None)

        # Find header row
        header_idx = None
        for i in range(min(10, len(df))):
            row_vals = df.iloc[i].astype(str).tolist()
            if 'Year' in row_vals and 'Country' in row_vals:
                header_idx = i
                break

        if header_idx is None:
            raise ValueError(f"Could not find header in agri-env sheet {sheet}")

        data = df.iloc[header_idx + 1:, :3].copy()
        data.columns = ['Year', 'Country', 'Area']
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
        data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
        data = data.dropna(subset=['Year'])
        data['Year'] = data['Year'].astype(int)

        # Aggregate: sum area across all countries per year
        agg = data.groupby('Year')['Area'].sum().reset_index()
        agg.columns = ['Year', col_name]
        results[col_name] = agg

    # Merge higher-level and entry-level
    result = results['Agri_Env_Higher_Area'].merge(
        results['Agri_Env_Entry_Area'], on='Year', how='outer'
    )
    result = result.sort_values('Year').reset_index(drop=True)
    print(f"  Agri-Environment: {len(result)} rows, years {result['Year'].min()}-{result['Year'].max()}")
    return result


def load_habitat_connectivity(filepath=None):
    """
    Load habitat connectivity (butterfly functional connectivity) data.

    Extracts the smoothed index from Sheet 1.

    Returns a DataFrame with columns: Year, Habitat_Connectivity
    """
    if filepath is None:
        filepath = os.path.join(_data_dir(), "UK-BDI-2025-habitat-connectivity.xlsx")

    df = pd.read_excel(filepath, sheet_name='1', engine='openpyxl', header=None)

    # Find header row containing 'Year' and 'Smoothed index'
    header_idx = None
    for i in range(min(10, len(df))):
        row_vals = df.iloc[i].astype(str).tolist()
        if any('Year' in str(v) for v in row_vals) and any('Smoothed' in str(v) for v in row_vals):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header row in habitat connectivity dataset")

    # Year is col 0, Smoothed index is col 2
    data = df.iloc[header_idx + 1:, [0, 2]].copy()
    data.columns = ['Year', 'Habitat_Connectivity']
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    data['Habitat_Connectivity'] = pd.to_numeric(data['Habitat_Connectivity'], errors='coerce')
    data = data.dropna(subset=['Year'])
    data['Year'] = data['Year'].astype(int)
    data = data.reset_index(drop=True)

    print(f"  Habitat Connectivity: {len(data)} rows, years {data['Year'].min()}-{data['Year'].max()}")
    return data


# ---------------------------------------------------------------------------
# Master merge
# ---------------------------------------------------------------------------

def merge_datasets():
    """
    Load all five datasets and merge them on Year using an outer join.

    Returns the merged DataFrame and a summary dict of missing-value counts.
    """
    print("Loading datasets...")
    pollinators = load_pollinating_insects()
    butterflies = load_butterflies()
    plants = load_plants()
    agri_env = load_agri_environment()
    connectivity = load_habitat_connectivity()

    print("\nMerging datasets on Year (outer join)...")
    master = pollinators.copy()

    for df in [butterflies, plants, agri_env, connectivity]:
        master = master.merge(df, on='Year', how='outer')

    master = master.sort_values('Year').reset_index(drop=True)

    # Add Year as a numeric feature for trend analysis
    master['Year_Numeric'] = master['Year']

    # Summarise missing values
    feature_cols = [c for c in master.columns if c != 'Year']
    missing_summary = {}
    for col in feature_cols:
        n_missing = master[col].isna().sum()
        if n_missing > 0:
            missing_years = master.loc[master[col].isna(), 'Year'].tolist()
            missing_summary[col] = {
                'count': n_missing,
                'years': missing_years
            }

    print(f"\nMerged dataset: {master.shape[0]} rows x {master.shape[1]} columns")
    print(f"Year range: {master['Year'].min()} - {master['Year'].max()}")
    print(f"\nMissing values per feature:")
    for col in feature_cols:
        n_miss = master[col].isna().sum()
        print(f"  {col}: {n_miss}/{len(master)} missing")

    return master, missing_summary


def interpolate_data(df, method='linear'):
    """
    Interpolate missing values in the dataframe using the specified method.

    Parameters
    ----------
    df : pd.DataFrame
        The merged dataframe with Year column and feature columns.
    method : str
        One of 'linear', 'polynomial', 'spline'.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values filled via interpolation.
    """
    result = df.copy()
    result = result.set_index('Year').sort_index()

    feature_cols = [c for c in result.columns if c not in ['Year_Numeric']]

    for col in feature_cols:
        if result[col].isna().any():
            if method == 'linear':
                result[col] = result[col].interpolate(method='linear')
            elif method == 'polynomial':
                try:
                    result[col] = result[col].interpolate(method='polynomial', order=2)
                except Exception:
                    result[col] = result[col].interpolate(method='linear')
            elif method == 'spline':
                try:
                    result[col] = result[col].interpolate(method='spline', order=3)
                except Exception:
                    result[col] = result[col].interpolate(method='linear')

    # Update Year_Numeric to match the index
    result['Year_Numeric'] = result.index

    result = result.reset_index()
    return result


# ---------------------------------------------------------------------------
# Main entry point (for standalone use)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    master_df, missing = merge_datasets()
    print("\n--- Merged Dataset Head ---")
    print(master_df.head(10).to_string())
    print("\n--- Merged Dataset Tail ---")
    print(master_df.tail(10).to_string())

    # Save raw merged dataset
    output_dir = os.path.join(_project_root(), 'data')
    os.makedirs(output_dir, exist_ok=True)
    master_df.to_csv(os.path.join(output_dir, 'cleaned_merged_dataset.csv'), index=False)
    print(f"\nSaved merged dataset to {output_dir}/cleaned_merged_dataset.csv")
