import numpy as np
from tqdm import tqdm  # For progress bar

def hamming_distance(v1, v2):
    """Calculate the Hamming distance between two vectors (number of differing positions)"""
    return np.sum(v1 != v2)

def simulate_vector_distinctness(N, D, min_distance=1, num_trials=1000):
    """
    Monte Carlo simulation to calculate the probability that N random vectors 
    in D dimensions are all distinct with at least min_distance positions different.
    
    Args:
        N: Number of vectors
        D: Number of dimensions
        min_distance: Minimum Hamming distance required between any pair of vectors
        num_trials: Number of Monte Carlo trials
    
    Returns:
        Probability estimate that all vectors have the minimum required distance
    """
    success_count = 0
    
    for _ in tqdm(range(num_trials)):
        # Generate N random vectors with D dimensions, each element is -1 or 1
        vectors = np.random.choice([-1, 1], size=(N, D))
        
        # Check minimum distance condition between all pairs
        all_meet_min_distance = True
        
        for i in range(N):
            for j in range(i+1, N):
                distance = hamming_distance(vectors[i], vectors[j])
                if distance < min_distance:
                    all_meet_min_distance = False
                    break
            if not all_meet_min_distance:
                break
        
        if all_meet_min_distance:
            success_count += 1
    
    probability = success_count / num_trials
    return probability

def main():
    # Example usage
    N = 50000  # Using smaller N for reasonable simulation time
    D_values = [40, 50, 60]
    P_values = [8]
    
    for P in P_values:
        print(f"\nRunning simulation with N={N} vectors, minimum distance P={P}")
        print("D\tProbability of meeting minimum distance")
        print("-" * 45)
        
        for D in D_values:
            prob = simulate_vector_distinctness(N, D, min_distance=P, num_trials=20)
            print(f"{D}\t{prob:.4f}")
    
    # For a single larger run
    print("\nEnter your own values:")
    try:
        user_N = int(input("Number of vectors (N): "))
        user_D = int(input("Number of dimensions (D): "))
        user_P = int(input("Minimum distance (P): "))
        user_trials = int(input("Number of trials (default 100): ") or "100")
        
        print(f"\nRunning with N={user_N}, D={user_D}, P={user_P}, trials={user_trials}")
        prob = simulate_vector_distinctness(user_N, user_D, min_distance=user_P, num_trials=user_trials)
        print(f"Probability of meeting minimum distance requirement: {prob:.4f}")
    except ValueError:
        print("Invalid input. Please enter integers.")

if __name__ == "__main__":
    main()