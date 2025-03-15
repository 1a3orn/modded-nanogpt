import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import math

def probability_all_vectors_min_distance(N, D, min_distance, success_probability=None):
    """
    Calculate the probability that all N random D-dimensional vectors have
    at least min_distance different dimensions between each pair.
    
    Args:
        N: Number of random vectors
        D: Dimension of each vector
        min_distance: Minimum Hamming distance required between any pair
        success_probability: If provided, calculates the required D for this probability
        
    Returns:
        If success_probability is None: probability that all vectors maintain minimum distance
        If success_probability is provided: minimum D required to achieve that probability
    """
    if success_probability is not None:
        # Calculate required D for given probability
        # We need to solve: 1 - N*(N-1)/2 * P(dist < min_dist) >= success_probability
        # Binary search for the minimum D that satisfies this
        low_D = min_distance
        high_D = 1000  # Some large upper bound
        
        while low_D <= high_D:
            mid_D = (low_D + high_D) // 2
            prob = 1 - (N * (N-1) / 2) * probability_distance_less_than(mid_D, min_distance)
            
            if prob >= success_probability:
                high_D = mid_D - 1
            else:
                low_D = mid_D + 1
        
        return low_D
    
    # Calculate probability that all pairs of vectors have distance >= min_distance
    # P(all pairs have dist >= min_dist) = 1 - P(at least one pair has dist < min_dist)
    # Using union bound: P(at least one pair has dist < min_dist) <= sum of P(pair_i has dist < min_dist)
    
    # Number of pairs
    num_pairs = comb(N, 2)
    
    # Probability that a specific pair has distance < min_distance
    p_too_close = probability_distance_less_than(D, min_distance)
    
    # Union bound for the probability that at least one pair is too close
    p_at_least_one_too_close = num_pairs * p_too_close
    
    # Probability that all pairs maintain minimum distance
    return max(0, 1 - p_at_least_one_too_close)

def probability_distance_less_than(D, min_distance):
    """
    Calculate the probability that two random D-dimensional binary vectors
    have a Hamming distance less than min_distance.
    """
    # For two random binary vectors, the probability distribution of their
    # Hamming distance follows a binomial distribution with p=0.5
    
    # Sum probabilities for distances 0, 1, 2, ..., min_distance-1
    p_too_close = 0
    for k in range(min_distance):
        p_too_close += comb(D, k) * (0.5)**D
    
    return p_too_close

def calculate_required_dimensions(N_values, min_distances, target_probabilities):
    """
    Calculate required dimensions for various combinations of parameters.
    
    Args:
        N_values: List of different N values to try
        min_distances: List of minimum distances to try
        target_probabilities: List of target success probabilities
        
    Returns:
        Dictionary with results for each combination
    """
    results = {}
    
    for N in N_values:
        results[N] = {}
        for min_dist in min_distances:
            results[N][min_dist] = {}
            for prob in target_probabilities:
                D = probability_all_vectors_min_distance(N, None, min_dist, prob)
                results[N][min_dist][prob] = D
    
    return results

def print_results_table(results):
    """Print results in a tabular format"""
    # Get unique values for each parameter
    N_values = sorted(results.keys())
    min_distances = sorted(list(results[N_values[0]].keys()))
    target_probabilities = sorted(list(results[N_values[0]][min_distances[0]].keys()))
    
    # Print header
    print("\nRequired dimensions (D) for different parameters:")
    print("-" * 80)
    
    for N in N_values:
        print(f"N = {N}")
        print("-" * 80)
        
        # Header row for probabilities
        header = "Min Distance |"
        for prob in target_probabilities:
            header += f" p={prob:.3f} |"
        print(header)
        print("-" * len(header))
        
        # Data rows
        for min_dist in min_distances:
            row = f"{min_dist:^11} |"
            for prob in target_probabilities:
                row += f" {results[N][min_dist][prob]:^8} |"
            print(row)
        print("-" * 80)
        print()

def plot_results(results):
    """Create plots of the results"""
    # Plot D vs. min_distance for different probabilities (fixed N)
    N = list(results.keys())[0]  # Use the first N value
    min_distances = sorted(list(results[N].keys()))
    target_probabilities = sorted(list(results[N][min_distances[0]].keys()))
    
    plt.figure(figsize=(10, 6))
    for prob in target_probabilities:
        D_values = [results[N][min_dist][prob] for min_dist in min_distances]
        plt.plot(min_distances, D_values, marker='o', label=f"p={prob:.3f}")
    
    plt.xlabel('Minimum Hamming Distance')
    plt.ylabel('Required Dimensions (D)')
    plt.title(f'Required Dimensions vs. Minimum Distance (N={N})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('dimensions_vs_distance.png')
    
    # If multiple N values, plot D vs. N for different min_distances (fixed prob)
    if len(results) > 1:
        N_values = sorted(results.keys())
        prob = target_probabilities[-1]  # Use the highest probability
        
        plt.figure(figsize=(10, 6))
        for min_dist in min_distances:
            D_values = [results[N][min_dist][prob] for N in N_values]
            plt.plot(N_values, D_values, marker='o', label=f"min_dist={min_dist}")
        
        plt.xlabel('Number of Vectors (N)')
        plt.ylabel('Required Dimensions (D)')
        plt.title(f'Required Dimensions vs. Number of Vectors (p={prob:.3f})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('dimensions_vs_vectors.png')

# Example usage
if __name__ == "__main__":
    # Parameters to explore
    N_values = [10000, 50000, 100000]
    min_distances = [1, 2, 4, 8, 16, 32]
    target_probabilities = [0.5, 0.9, 0.99, 0.999, 0.99999]
    
    # Calculate results
    print("Calculating required dimensions. This may take a moment...")
    results = calculate_required_dimensions(N_values, min_distances, target_probabilities)
    
    # Print results
    print_results_table(results)
    
    # Create additional analysis functions if needed
    def theoretical_analysis():
        """Print theoretical analysis information"""
        print("\nTheoretical Analysis:")
        print("-" * 60)
        
        N = 50000
        print(f"For N = {N} vectors:")
        
        for min_dist in [1, 2, 4, 8]:
            p_bound = 0.99
            
            # For min_dist = 1, we need all vectors to be distinct
            if min_dist == 1:
                D = math.ceil(math.log2(N*(N-1)/(2*(1-p_bound))))
                explanation = "vectors must be distinct"
            else:
                # For min_dist > 1, we need volume of hamming balls to be considered
                # This is a simplification using volume bound
                volume_factor = sum(comb(D, k) for k in range(min_dist//2))
                D = math.ceil(math.log2(N) + math.log2(volume_factor))
                explanation = f"using volume bound with radius {min_dist//2}"
            
            print(f"Min distance = {min_dist}: D ≈ {D} ({explanation})")
        
        print("-" * 60)
    
    # Uncomment to include theoretical analysis
    # theoretical_analysis()
    
    # Create plots (uncomment if matplotlib is available)
    # plot_results(results)
    
    # Specific case from the original question
    N = 50000
    print(f"\nSpecific case: N = {N}, minimum distance = 1")
    for prob in [0.5, 0.99, 0.999, 0.9999]:
        D = probability_all_vectors_min_distance(N, None, 1, prob)
        print(f"For {prob*100:.1f}% probability: D = {D}")