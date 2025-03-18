#!/usr/bin/env python3
"""
Run All N-Body Experiments

This script runs all the n-body experiments:
1. Sun-Jupiter-Saturn three-body system
2. Full Solar System
3. TRAPPIST-1 system

It provides options to select which experiments to run.
"""
import sys
import argparse
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run N-body experiments")
    
    parser.add_argument("--all", action="store_true", 
                        help="Run all experiments")
    
    parser.add_argument("--three-body", action="store_true", 
                        help="Run the Sun-Jupiter-Saturn three-body experiment")
    
    parser.add_argument("--solar-system", action="store_true", 
                        help="Run the full Solar System experiment")
    
    parser.add_argument("--trappist1", action="store_true", 
                        help="Run the TRAPPIST-1 system experiment")
    
    parser.add_argument("--integrators", type=str, nargs="+", 
                        choices=["all", "euler", "leapfrog", "rk4", "wisdom_holman"],
                        default=["all"], 
                        help="Integrators to use (default: all)")
    
    parser.add_argument("--circular-only", action="store_true", 
                        help="Run only circular orbit experiments (not eccentric)")
    
    args = parser.parse_args()
    
    # If no specific experiment is selected, run all
    if not (args.three_body or args.solar_system or args.trappist1 or args.all):
        args.all = True
    
    return args

def main():
    """Run the selected experiments."""
    args = parse_arguments()
    
    # Determine which experiments to run
    run_three_body = args.all or args.three_body
    run_solar_system = args.all or args.solar_system
    run_trappist1 = args.all or args.trappist1
    
    # Determine which integrators to use
    if "all" in args.integrators:
        integrators = ["euler", "leapfrog", "rk4", "wisdom_holman"]
    else:
        integrators = args.integrators
    
    # Print experiment settings
    print("\nRunning N-body experiments with the following settings:")
    print(f"  Experiments:")
    print(f"    - Sun-Jupiter-Saturn: {'Yes' if run_three_body else 'No'}")
    print(f"    - Full Solar System:  {'Yes' if run_solar_system else 'No'}")
    print(f"    - TRAPPIST-1 System:  {'Yes' if run_trappist1 else 'No'}")
    print(f"  Integrators: {', '.join(integrators)}")
    print(f"  Orbit types: {'Circular only' if args.circular_only else 'Circular and eccentric'}")
    print("\n" + "="*50 + "\n")
    
    if run_three_body:
        print("Running Sun-Jupiter-Saturn experiment...")
        from experiments.run_sun_jupiter_saturn import main as run_three_body_main
        run_three_body_main()
        print("\n" + "="*50 + "\n")
    
    if run_solar_system:
        print("Running Solar System experiment...")
        from experiments.run_solar_system import main as run_solar_system_main
        run_solar_system_main()
        print("\n" + "="*50 + "\n")
    
    if run_trappist1:
        print("Running TRAPPIST-1 system experiment...")
        from experiments.run_trappist1 import main as run_trappist1_main
        run_trappist1_main()
        print("\n" + "="*50 + "\n")
    
    print("All experiments completed!")

if __name__ == "__main__":
    main() 