import matplotlib.pyplot as plt
import json
import sys
import os
from pathlib import Path
import numpy as np
from slip import main
from visualize import animate_slip

def load_test_cases(filename="test_cases.json"):
    try:
        with open(filename, 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Config file {filename} not found. Using default test cases.")
        return [
            {"name": "baseline", "m": 10.0, "h0": 2.0, "g": 9.81, "k_initial": 200.0, "k_final": 400.0, "h_max": 3.0},
            
        ]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filename}: {e}")

    required_keys = ["name", "m", "h0", "g", "k_initial", "k_final", "h_max"]
    for i, test in enumerate(test_cases):
        missing = [key for key in required_keys if key not in test]
        if missing:
            raise ValueError(f"Test case {i} missing required keys: {missing}")

    return test_cases

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "test_cases.json"
    test_cases = load_test_cases(config_file)

    base_dir = Path("experiments")
    base_dir.mkdir(exist_ok=True)

    print("Running test cases...")
    all_results = []

    for test in test_cases:
        print(f"\n{test['name']}: m={test['m']}, k={test['k_initial']}->{test['k_final']}, h_max={test['h_max']}")

        exp_dir = base_dir / test['name']
        exp_dir.mkdir(exist_ok=True)

        times, states = main(
            mass=test["m"],
            spring_length=test["h0"],
            gravity=test["g"],
            k_initial=test["k_initial"],
            k_final=test["k_final"],
            initial_height=test["h_max"]
        )

        all_results.append({
            "name": test["name"],
            "times": times,
            "states": states,
            "params": test
        })

        print(f"  Saving phase plot...")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(states[:, 0], states[:, 1], linewidth=1.5)
        ax.set_xlabel('Position (h)')
        ax.set_ylabel('Velocity (h_dot)')
        ax.set_title(f"{test['name']}: m={test['m']}, k={test['k_initial']}->{test['k_final']}")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(exp_dir / "phase_plot.png", dpi=150)
        plt.close()

        print(f"  Saving animation...")
        animate_slip(
            times,
            states,
            test["h0"],
            realtime_factor=1.0,
            save_file=str(exp_dir / "animation.mp4"),
            show=False
        )

        print(f"  Saved to {exp_dir}/")

    print("\n\nGenerating comparison plot...")
    n_cases = len(all_results)
    n_cols = 3
    n_rows = int(np.ceil(n_cases / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_cases == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, result in enumerate(all_results):
        ax = axes[idx]
        ax.plot(result["states"][:, 0], result["states"][:, 1], linewidth=1.5)
        ax.set_xlabel('Position (h)')
        ax.set_ylabel('Velocity (h_dot)')
        ax.set_title(result["name"])
        ax.grid(True)

    for idx in range(len(all_results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(base_dir / 'comparison.png', dpi=150)
    plt.close()
    print(f"Saved comparison to {base_dir / 'comparison.png'}")

    print(f"\nAll experiments saved to {base_dir}/")
