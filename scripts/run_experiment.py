import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime

# Dodaj katalog główny projektu do ścieżek systemowych
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from src.experiments.convergence_analysis import run_convergence_analysis
from src.experiments.parameter_analysis import run_parameter_analysis
from src.experiments.size_analysis import run_size_analysis
from src.experiments.structure_analysis import run_structure_analysis
from src.experiments.hybrid_analysis import run_hybrid_analysis

def main():
    """Główny punkt wejścia do przeprowadzenia eksperymentów."""
    parser = argparse.ArgumentParser(description='TSP Algorithm Analysis Framework')
    
    # Argumenty dotyczące eksperymentu
    parser.add_argument('--experiment', type=str, default='all',
                      choices=['all', 'convergence', 'parameters', 'size', 'structure', 'hybrid'],
                      help='Type of experiment to run')
    
    # Argumenty dotyczące zakresu eksperymentów
    parser.add_argument('--quick', action='store_true',
                      help='Run a quick test with reduced problem sizes and repetitions')
    parser.add_argument('--sizes', type=str, default=None,
                      help='Comma-separated list of instance sizes to test (e.g., "10,20,50")')
    parser.add_argument('--runs', type=int, default=3,
                      help='Number of runs per instance configuration')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Argumenty dotyczące katalogu wyników
    parser.add_argument('--output', type=str, default='data/results',
                      help='Base output directory for results')
    parser.add_argument('--timestamp', action='store_true',
                      help='Add timestamp to output directory')
    
    args = parser.parse_args()
    
    # Ustal rozmiary instancji
    if args.sizes:
        sizes = [int(size) for size in args.sizes.split(',')]
    elif args.quick:
        sizes = [10, 20, 30]  # Mniejsze instancje dla szybkich testów
    else:
        sizes = [10, 20, 30, 50, 75, 100]
    
    # Dostosuj liczbę uruchomień dla szybkich testów
    runs = 2 if args.quick else args.runs
    
    # Utwórz katalog wynikowy
    output_dir = args.output
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Ustaw ziarno generatora liczb losowych
    np.random.seed(args.seed)
    
    # Uruchom wybrane eksperymenty
    start_time = time.time()
    
    if args.experiment in ['all', 'convergence']:
        print("\n=== Running Convergence Analysis ===")
        convergence_sizes = [s for s in sizes if s >= 20][:2]  # Wybierz max 2 średnie/duże rozmiary
        run_convergence_analysis(
            output_dir=os.path.join(output_dir, 'convergence'),
            sizes=convergence_sizes,
            runs=runs,
            seed=args.seed
        )
    
    if args.experiment in ['all', 'parameters']:
        print("\n=== Running Parameter Analysis ===")
        param_size = [s for s in sizes if s >= 20][0]  # Wybierz pierwszy średni/duży rozmiar
        run_parameter_analysis(
            output_dir=os.path.join(output_dir, 'parameters'),
            size=param_size,
            runs=runs,
            seed=args.seed
        )
    
    if args.experiment in ['all', 'size']:
        print("\n=== Running Size Analysis ===")
        run_size_analysis(
            output_dir=os.path.join(output_dir, 'size'),
            sizes=sizes,
            runs=runs,
            seed=args.seed
        )
    
    if args.experiment in ['all', 'structure']:
        print("\n=== Running Structure Analysis ===")
        structure_sizes = [min(sizes), max(s for s in sizes if s <= 50)]
        run_structure_analysis(
            output_dir=os.path.join(output_dir, 'structure'),
            sizes=structure_sizes,
            runs=runs,
            seed=args.seed
        )
    
    if args.experiment in ['all', 'hybrid']:
        print("\n=== Running Hybrid Algorithm Analysis ===")
        hybrid_sizes = [s for s in sizes if s >= 20][:3]  # Wybierz max 3 średnie/duże rozmiary
        run_hybrid_analysis(
            output_dir=os.path.join(output_dir, 'hybrid'),
            sizes=hybrid_sizes,
            runs=runs,
            seed=args.seed
        )
    
    elapsed_time = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()