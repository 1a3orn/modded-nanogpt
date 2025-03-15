import numpy as np
from tqdm import tqdm  # For progress bar

def simulate_vector_distinctness(N, D, num_trials=100):
    """
    Monte Carlo simulation to calculate the probability that N random vectors 
    in D dimensions are all distinct.
    
    Args:
        N: Number of vectors
        D: Number of dimensions
        num_trials: Number of Monte Carlo trials
    
    Returns:
        Probability estimate that all vectors are distinct
    """
    success_count = 0
    
    for _ in tqdm(range(num_trials)):
        # Generate N random vectors with D dimensions, each element is -1 or 1
        vectors = np.random.choice([-1, 1], size=(N, D))
        
        # Check if all vectors are distinct
        # Convert each vector to a tuple for hashing
        vector_set = set(tuple(v) for v in vectors)
        
        # If the set has N elements, all vectors are distinct
        if len(vector_set) == N:
            success_count += 1
    
    probability = success_count / num_trials
    return probability

def main():
    # Example usage
    N = 50000  # Smaller N for reasonable simulation time
    D_values = [30, 40, 50]
    
    print(f"Running simulation with N={N} vectors, {1000} trials per D value")
    print("D\tProbability of all vectors being distinct")
    print("-" * 45)
    
    for D in D_values:
        prob = simulate_vector_distinctness(N, D)
        print(f"{D}\t{prob:.4f}")
    
    # For a single larger run
    print("\nEnter your own values:")
    try:
        user_N = int(input("Number of vectors (N): "))
        user_D = int(input("Number of dimensions (D): "))
        user_trials = int(input("Number of trials (default 100): ") or "100")
        
        print(f"\nRunning with N={user_N}, D={user_D}, trials={user_trials}")
        prob = simulate_vector_distinctness(user_N, user_D, user_trials)
        print(f"Probability of all vectors being distinct: {prob:.4f}")
    except ValueError:
        print("Invalid input. Please enter integers.")

if __name__ == "__main__":
    main()