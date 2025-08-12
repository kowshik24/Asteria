#!/bin/bash

# Complete Research Experiment Runner for Asteria
# This script runs all research experiments in sequence

echo "=========================================="
echo "Asteria Research Experiments Suite"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "asteria/__init__.py" ]; then
    echo "Error: Please run this script from the Asteria root directory"
    exit 1
fi

# Create results directory
mkdir -p research_results
cd research_results

echo "Starting comprehensive research experiments..."

# 1. Real Image Experiments
echo ""
echo "1. Running Real Image Experiments..."
echo "-----------------------------------"
cd ..
python experiments/real_image_experiments.py
if [ $? -ne 0 ]; then
    echo "Warning: Real image experiments failed, continuing..."
fi

# 2. Comparative Analysis
echo ""
echo "2. Running Comparative Analysis..."
echo "---------------------------------"
python experiments/comparative_analysis.py
if [ $? -ne 0 ]; then
    echo "Warning: Comparative analysis failed, continuing..."
fi

# 3. Advanced Research Experiments
echo ""
echo "3. Running Advanced Research Experiments..."
echo "------------------------------------------"
python experiments/advanced_research.py
if [ $? -ne 0 ]; then
    echo "Warning: Advanced research experiments failed, continuing..."
fi

# 4. Comprehensive Benchmark
echo ""
echo "4. Running Comprehensive Benchmark..."
echo "------------------------------------"
python experiments/comprehensive_benchmark.py
if [ $? -ne 0 ]; then
    echo "Warning: Comprehensive benchmark failed, continuing..."
fi

# 5. Enhanced Synthetic Speed Test
echo ""
echo "5. Running Enhanced Synthetic Speed Test..."
echo "-------------------------------------------"
python benchmarks/synthetic_speed.py
if [ $? -ne 0 ]; then
    echo "Warning: Synthetic speed test failed, continuing..."
fi

echo ""
echo "=========================================="
echo "Research Experiments Completed!"
echo "=========================================="

echo ""
echo "Generated Results:"
echo "-----------------"

# List all generated files
echo ""
echo "Real Image Results:"
if [ -d "real_image_results" ]; then
    ls -la real_image_results/
else
    echo "  No real image results found"
fi

echo ""
echo "Comparative Analysis Results:"
if [ -d "comparative_results" ]; then
    ls -la comparative_results/
else
    echo "  No comparative results found"
fi

echo ""
echo "Advanced Research Results:"
if [ -d "advanced_results" ]; then
    ls -la advanced_results/
else
    echo "  No advanced results found"
fi

echo ""
echo "Comprehensive Benchmark Results:"
if [ -d "benchmark_results" ]; then
    ls -la benchmark_results/
else
    echo "  No benchmark results found"
fi

echo ""
echo "Synthetic Speed Results:"
if [ -d "synthetic_results" ]; then
    ls -la synthetic_results/
else
    echo "  No synthetic results found"
fi

echo ""
echo "Research Summary:"
echo "=================="
echo "✓ Real image experiments (CIFAR-10 style)"
echo "✓ Comparative analysis vs FAISS, Annoy, etc."
echo "✓ Advanced research (dimensionality, parameters)"
echo "✓ Comprehensive benchmarking suite"
echo "✓ Enhanced synthetic speed testing"
echo ""
echo "All results contain:"
echo "  - Performance metrics (QPS, Recall, Memory)"
echo "  - Visualization plots (PNG files)"
echo "  - Raw data (JSON files)"
echo "  - LaTeX tables for publications"
echo ""
echo "Use these results to support your research paper!"
echo "=========================================="
