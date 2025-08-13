#!/bin/bash

# Fast Research Experiment Runner for Asteria
# This script runs experiments with smaller datasets for quick testing

echo "=========================================="
echo "Asteria FAST Research Experiments Suite"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "asteria/__init__.py" ]; then
    echo "Error: Please run this script from the Asteria root directory"
    exit 1
fi

# Create results directory
mkdir -p research_results_fast
cd research_results_fast

echo "Starting FAST research experiments with smaller datasets..."

# 1. Fast Real Image Experiments (smaller dataset)
echo ""
echo "1. Running Fast Real Image Experiments..."
echo "-----------------------------------"
cd ..
python -c "
import sys
import os
sys.path.insert(0, '.')

from experiments.real_image_experiments import RealImageExperiment

# Create experiment with smaller datasets
experiment = RealImageExperiment('research_results_fast/real_image_results')

print('Running FAST CIFAR-10 experiment with 2000 samples...')
# Run with much smaller dataset
experiment.run_cifar10_experiment(subset_size=2000, feature_model='resnet')

print('Running FAST scalability study...')
# Run with smaller max size
experiment.run_scalability_study(max_db_size=10000)

# Save results
experiment.save_results()
experiment.generate_summary()
print('Fast real image experiments completed!')
"

if [ $? -ne 0 ]; then
    echo "Warning: Fast real image experiments failed, continuing..."
fi

# 2. Fast Comprehensive Benchmark
echo ""
echo "2. Running Fast Comprehensive Benchmark..."
echo "------------------------------------"
python -c "
import sys
import os
sys.path.insert(0, '.')

from experiments.comprehensive_benchmark import ComprehensiveBenchmark

# Create benchmark with smaller datasets
benchmark = ComprehensiveBenchmark('research_results_fast/benchmark_results')

print('Running fast scale experiment...')
# Much smaller scale experiment
benchmark.run_scale_experiment(
    db_sizes=[500, 1000, 2500, 5000],
    query_size=200,
    dim=512
)

print('Running fast parameter study...')
# Simplified parameter study with valid configurations
base_config = {
    'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48,
    'rank': 48, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
}

# Use only configurations that ensure rank is divisible by blocks
param_ranges = {
    'code_bits': [32, 48, 64],  # Ensure code_bits >= raw_bits
    'm_vantages': [36, 48, 60],  # All divisible by 12
    'max_radius': [1, 2, 3]
}

benchmark.run_parameter_study(base_config, param_ranges, db_size=5000, query_size=200)

print('Running fast memory vs accuracy study...')
benchmark.run_memory_vs_accuracy(db_size=5000, query_size=200)

# Save results
benchmark.save_results()
benchmark.generate_summary_report()
print('Fast benchmark completed!')
"

if [ $? -ne 0 ]; then
    echo "Warning: Fast benchmark failed, continuing..."
fi

# 3. Fast Synthetic Speed Test
echo ""
echo "3. Running Fast Synthetic Speed Test..."
echo "-------------------------------------------"
python -c "
import sys
import os
sys.path.insert(0, '.')

from benchmarks.synthetic_speed import SyntheticBenchmark

# Create benchmark with smaller datasets
benchmark = SyntheticBenchmark('research_results_fast/synthetic_results')

print('Running fast scale benchmark...')
# Run scale benchmark with smaller datasets
benchmark.run_scale_benchmark(
    db_sizes=[1000, 2500, 5000],
    query_size=200,
    dim=512
)

print('Running fast parameter sweep...')
# Run parameter sweep
benchmark.run_parameter_sweep(
    db_size=2500,
    query_size=200,
    dim=512
)

print('Running fast dimension study...')
# Run dimension study
benchmark.run_dimension_study(
    dimensions=[256, 512],
    db_size=2500,
    query_size=200
)

print('Fast synthetic benchmark completed!')
"

if [ $? -ne 0 ]; then
    echo "Warning: Fast synthetic test failed, continuing..."
fi

echo ""
echo "=========================================="
echo "Fast Research Experiments Completed!"
echo "=========================================="

echo ""
echo "Generated Results (in research_results_fast/):"
echo "---------------------------------------------"

# List all generated files
echo ""
if [ -d "research_results_fast" ]; then
    echo "Fast Results Summary:"
    find research_results_fast -name "*.png" -o -name "*.json" -o -name "*.txt" | sort
    echo ""
    echo "File count:"
    find research_results_fast -type f | wc -l
    echo " files generated"
else
    echo "  No fast results found"
fi

echo ""
echo "Fast Research Summary:"
echo "======================"
echo "✓ Fast CIFAR-10 experiments (2K samples)"
echo "✓ Fast scalability study (up to 10K)"
echo "✓ Fast comprehensive benchmarking"
echo "✓ Fast synthetic speed testing"
echo ""
echo "Results contain essential metrics for quick validation:"
echo "  - Performance metrics (QPS, Recall, Memory)"
echo "  - Key visualization plots"
echo "  - Summary statistics"
echo ""
echo "Use these for rapid prototyping and initial validation!"
echo "Run ./run_all_experiments.sh for full-scale experiments."
echo "=========================================="
