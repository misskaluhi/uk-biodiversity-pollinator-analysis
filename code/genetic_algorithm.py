"""
genetic_algorithm.py
====================
Implements a Genetic Algorithm (GA) to jointly optimise:
  1. Feature selection   – which of the 10 candidate features to include
  2. Year range          – start and end year for the analysis window
  3. Interpolation method – how to fill missing values (linear / polynomial / spline)

The fitness function trains a Random Forest regressor on the selected
configuration and returns the R² score via 5-fold cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

# ---------------------------------------------------------------------------
# Feature and encoding definitions
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    'Butterfly_All_Species',      # F1
    'Butterfly_Specialists',      # F2
    'Butterfly_Generalists',      # F3
    'Plants_Arable',              # F4
    'Plants_Lowland_Grassland',   # F5
    'Plants_Woodland',            # F6
    'Agri_Env_Higher_Area',       # F7
    'Agri_Env_Entry_Area',        # F8
    'Habitat_Connectivity',       # F9
    'Year_Numeric',               # F10  (year as trend proxy)
]

INTERPOLATION_METHODS = ['linear', 'polynomial', 'spline']

TARGET_COL = 'Pollinator_Index'

# Year bounds for the target variable (pollinator data)
MIN_YEAR = 1980
MAX_YEAR = 2024
NUM_YEARS = MAX_YEAR - MIN_YEAR + 1  # 45


# ---------------------------------------------------------------------------
# Chromosome structure
# ---------------------------------------------------------------------------
# [F1, F2, ..., F10, Y_start (6 bits), Y_end (6 bits), Interp (2 bits)]
#
# Total bits = 10 + 6 + 6 + 2 = 24
#
# Y_start and Y_end are encoded as 6-bit integers (0-63) mapped linearly
# to the year range 1980-2024.
# Interp: 00 = linear, 01 = polynomial, 10 = spline, 11 = linear (wrap)

NUM_FEATURES = 10
YEAR_BITS = 6
INTERP_BITS = 2
CHROMOSOME_LENGTH = NUM_FEATURES + 2 * YEAR_BITS + INTERP_BITS  # 24


def _bits_to_int(bits):
    """Convert a list of binary digits to an integer."""
    return int(''.join(str(b) for b in bits), 2)


def decode_chromosome(chromosome):
    """
    Decode a binary chromosome into its three components.

    Returns
    -------
    selected_features : list[str]
        Names of selected features.
    year_start : int
        Start year for the analysis window.
    year_end : int
        End year for the analysis window.
    interp_method : str
        Interpolation method name.
    """
    # Feature selection bits
    feature_bits = chromosome[:NUM_FEATURES]
    selected = [FEATURE_NAMES[i] for i, b in enumerate(feature_bits) if b == 1]

    # Year range bits
    y_start_bits = chromosome[NUM_FEATURES : NUM_FEATURES + YEAR_BITS]
    y_end_bits = chromosome[NUM_FEATURES + YEAR_BITS : NUM_FEATURES + 2 * YEAR_BITS]

    y_start_val = _bits_to_int(y_start_bits)
    y_end_val = _bits_to_int(y_end_bits)

    # Map to actual years (within pollinator data range 1980-2024)
    year_start = MIN_YEAR + int(y_start_val * (NUM_YEARS - 1) / (2**YEAR_BITS - 1))
    year_end = MIN_YEAR + int(y_end_val * (NUM_YEARS - 1) / (2**YEAR_BITS - 1))

    year_start = max(MIN_YEAR, min(year_start, MAX_YEAR))
    year_end = max(MIN_YEAR, min(year_end, MAX_YEAR))

    # Ensure start < end with a minimum window of 10 years
    if year_start >= year_end:
        year_start, year_end = min(year_start, year_end), max(year_start, year_end)
    if year_end - year_start < 10:
        year_end = min(year_start + 15, MAX_YEAR)
        year_start = max(year_end - 15, MIN_YEAR)

    # Interpolation method
    interp_bits = chromosome[NUM_FEATURES + 2 * YEAR_BITS:]
    interp_val = _bits_to_int(interp_bits) % 3
    interp_method = INTERPOLATION_METHODS[interp_val]

    return selected, year_start, year_end, interp_method


# ---------------------------------------------------------------------------
# Pre-interpolated data cache
# ---------------------------------------------------------------------------

def prepare_interpolated_datasets(master_df):
    """
    Pre-compute interpolated versions of the master dataframe for each
    interpolation method.  This avoids repeated interpolation during
    GA fitness evaluation.

    The strategy:
      1. Interpolate interior gaps using the chosen method.
      2. Forward-fill then backward-fill to handle leading/trailing NaN
         (extends the nearest known value to the edges).

    Returns a dict mapping method name -> interpolated DataFrame.
    """
    datasets = {}
    feature_cols = [c for c in master_df.columns if c not in ['Year']]

    for method in INTERPOLATION_METHODS:
        df = master_df.copy().set_index('Year').sort_index()

        for col in feature_cols:
            if df[col].isna().any():
                if method == 'linear':
                    df[col] = df[col].interpolate(method='linear')
                elif method == 'polynomial':
                    try:
                        df[col] = df[col].interpolate(method='polynomial', order=2)
                    except Exception:
                        df[col] = df[col].interpolate(method='linear')
                elif method == 'spline':
                    try:
                        df[col] = df[col].interpolate(method='spline', order=3)
                    except Exception:
                        df[col] = df[col].interpolate(method='linear')

                # Edge fill: extend nearest value to boundaries
                df[col] = df[col].ffill().bfill()

        datasets[method] = df.reset_index()

    return datasets


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def fitness_function(chromosome, interp_datasets, random_state=42):
    """
    Evaluate a chromosome by training a Random Forest and returning
    the mean 5-fold cross-validation R² score.

    Parameters
    ----------
    chromosome : list[int]
        Binary chromosome of length CHROMOSOME_LENGTH.
    interp_datasets : dict
        Pre-interpolated datasets keyed by interpolation method name.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    float
        Fitness score (mean cross-validated R²). Returns -1.0 for
        invalid configurations.
    """
    selected, year_start, year_end, interp_method = decode_chromosome(chromosome)

    # Penalty: no features selected
    if len(selected) == 0:
        return -1.0

    # Get the pre-interpolated dataset for this method
    df = interp_datasets[interp_method]

    # Filter to selected year range
    subset = df[(df['Year'] >= year_start) & (df['Year'] <= year_end)].copy()

    # Check target availability
    subset = subset.dropna(subset=[TARGET_COL])

    # Select features that actually exist as columns
    features_available = [f for f in selected if f in subset.columns]
    if len(features_available) == 0:
        return -1.0

    # Drop rows with NaN in selected features
    subset = subset[[TARGET_COL] + features_available].dropna()

    if len(subset) < 10:
        return -1.0

    X = subset[features_available].values
    y = subset[TARGET_COL].values

    # Drop zero-variance features (e.g. extrapolated constant columns)
    feature_vars = np.var(X, axis=0)
    nonzero_mask = feature_vars > 1e-10
    if not np.any(nonzero_mask):
        return -1.0
    X = X[:, nonzero_mask]

    # Train Random Forest with cross-validation
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    )

    n_folds = min(5, len(X))
    if n_folds < 3:
        return -1.0

    try:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
        mean_r2 = float(np.nanmean(scores))
        if np.isnan(mean_r2):
            return -1.0
        return mean_r2
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def initialise_population(pop_size, random_state=42):
    """Create an initial random population of binary chromosomes."""
    rng = np.random.RandomState(random_state)
    return [rng.randint(0, 2, CHROMOSOME_LENGTH).tolist() for _ in range(pop_size)]


def tournament_selection(population, fitness_scores, k=3, rng=None):
    """Select one individual via tournament selection (size k)."""
    if rng is None:
        rng = np.random.RandomState()
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
    return population[best_idx][:]


def crossover(parent1, parent2, crossover_rate=0.8, rng=None):
    """Single-point crossover."""
    if rng is None:
        rng = np.random.RandomState()
    if rng.random() < crossover_rate:
        point = rng.randint(1, CHROMOSOME_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1[:], parent2[:]


def mutate(chromosome, mutation_rate=0.1, rng=None):
    """Bit-flip mutation."""
    if rng is None:
        rng = np.random.RandomState()
    mutated = chromosome[:]
    for i in range(CHROMOSOME_LENGTH):
        if rng.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------

def run_ga(master_df, pop_size=50, n_generations=100, crossover_rate=0.8,
           mutation_rate=0.1, tournament_k=3, random_state=42, verbose=True):
    """
    Execute the Genetic Algorithm optimisation.

    Parameters
    ----------
    master_df : pd.DataFrame
        Full merged dataset.
    pop_size : int
        Population size.
    n_generations : int
        Number of generations.
    crossover_rate : float
        Probability of crossover.
    mutation_rate : float
        Per-bit mutation probability.
    tournament_k : int
        Tournament selection size.
    random_state : int
        Random seed.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    best_chromosome : list[int]
        The best chromosome found.
    best_fitness : float
        Fitness of the best chromosome.
    history : pd.DataFrame
        Generation-by-generation fitness history.
    """
    rng = np.random.RandomState(random_state)

    # Pre-compute interpolated datasets for each method
    if verbose:
        print("  Pre-computing interpolated datasets...")
    interp_datasets = prepare_interpolated_datasets(master_df)

    # Initialise population
    population = initialise_population(pop_size, random_state)

    # Track history
    history_records = []
    best_overall_fitness = -np.inf
    best_overall_chromosome = None

    for gen in range(n_generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for chrom in population:
            f = fitness_function(chrom, interp_datasets, random_state)
            fitness_scores.append(f)

        # Track statistics (use valid scores for mean/std)
        valid_scores = [s for s in fitness_scores if s > -1.0]
        gen_best = max(fitness_scores)
        gen_mean = np.mean(valid_scores) if valid_scores else -1.0
        gen_std = np.std(valid_scores) if valid_scores else 0.0
        gen_best_idx = int(np.argmax(fitness_scores))

        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
            best_overall_chromosome = population[gen_best_idx][:]

        history_records.append({
            'Generation': gen + 1,
            'Best_Fitness': gen_best,
            'Mean_Fitness': gen_mean,
            'Std_Fitness': gen_std,
            'Overall_Best': best_overall_fitness
        })

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            decoded = decode_chromosome(population[gen_best_idx])
            print(f"  Gen {gen+1:3d}/{n_generations}: "
                  f"Best R²={gen_best:.4f} | Mean={gen_mean:.4f} | "
                  f"Features={len(decoded[0])}, Years={decoded[1]}-{decoded[2]}, "
                  f"Interp={decoded[3]}")

        # Create next generation via selection, crossover, mutation
        new_population = []

        # Elitism: keep the best individual
        if best_overall_chromosome is not None:
            new_population.append(best_overall_chromosome[:])

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitness_scores, tournament_k, rng)
            p2 = tournament_selection(population, fitness_scores, tournament_k, rng)
            c1, c2 = crossover(p1, p2, crossover_rate, rng)
            c1 = mutate(c1, mutation_rate, rng)
            c2 = mutate(c2, mutation_rate, rng)
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population

    history_df = pd.DataFrame(history_records)

    if verbose:
        selected, y_start, y_end, interp = decode_chromosome(best_overall_chromosome)
        print(f"\n  GA Optimisation Complete!")
        print(f"  Best fitness (R²): {best_overall_fitness:.4f}")
        print(f"  Selected features: {selected}")
        print(f"  Year range: {y_start} - {y_end}")
        print(f"  Interpolation: {interp}")

    return best_overall_chromosome, best_overall_fitness, history_df


# ---------------------------------------------------------------------------
# Convenience: apply best solution to data
# ---------------------------------------------------------------------------

def apply_best_solution(master_df, chromosome):
    """
    Given the best chromosome and the master dataframe, prepare the
    final analysis-ready dataset.

    Returns
    -------
    final_df : pd.DataFrame
        Filtered, interpolated data with selected features and target.
    selected_features : list[str]
        Names of the selected features.
    config : dict
        Configuration details (year range, interpolation, etc.).
    """
    selected, year_start, year_end, interp_method = decode_chromosome(chromosome)

    config = {
        'year_start': year_start,
        'year_end': year_end,
        'selected_features': selected,
        'interpolation_method': interp_method
    }

    # Pre-compute interpolated dataset with the chosen method
    interp_datasets = prepare_interpolated_datasets(master_df)
    df = interp_datasets[interp_method]

    # Filter year range
    subset = df[
        (df['Year'] >= year_start) & (df['Year'] <= year_end)
    ].copy()

    # Drop rows where target is NaN
    subset = subset.dropna(subset=[TARGET_COL])

    # Keep only target + selected features that exist
    available = [f for f in selected if f in subset.columns]
    subset = subset[['Year', TARGET_COL] + available].dropna()
    subset = subset.reset_index(drop=True)

    config['selected_features'] = available
    config['final_rows'] = len(subset)

    return subset, available, config


# ---------------------------------------------------------------------------
# Main entry point (standalone testing)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from data_preprocessing import merge_datasets

    master_df, _ = merge_datasets()
    print("\nRunning Genetic Algorithm optimisation...")
    best_chrom, best_fit, history = run_ga(
        master_df, pop_size=50, n_generations=100,
        crossover_rate=0.8, mutation_rate=0.1, random_state=42
    )

    final_df, features, config = apply_best_solution(master_df, best_chrom)
    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Config: {config}")
