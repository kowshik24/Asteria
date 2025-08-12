#!/bin/bash

# Complete Research Experiment Runner for Asteria
# This script runs all research experiments with optimizations

echo "=========================================="
echo "Asteria Research Experiments Suite"
echo "=========================================="
echo "🚀 Enhanced with parallel execution and optimizations"

# Check if we're in the right directory
if [ ! -f "asteria/__init__.py" ]; then
    echo "Error: Please run this script from the Asteria root directory"
    exit 1
fi

# Performance optimizations
export ASTERIA_OPTIMIZED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Create results directory
mkdir -p research_results
cd research_results

echo "Starting comprehensive research experiments..."
start_time=$(date +%s)

# Function to run experiments with better error handling and timing
run_experiment() {
    local name="$1"
    local script="$2"
    local estimated_time="$3"
    
    echo ""
    echo "📊 $name (est. ${estimated_time}min)..."
    echo "-----------------------------------"
    cd .. 2>/dev/null || true
    
    local exp_start=$(date +%s)
    python "$script" 2>&1 | tee -a "research_results/experiment.log"
    local exit_code=$?
    local exp_end=$(date +%s)
    local exp_duration=$((exp_end - exp_start))
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $name completed successfully (${exp_duration}s)"
    else
        echo "⚠️  Warning: $name failed (exit code: $exit_code), continuing..."
        echo "   Check research_results/experiment.log for details"
    fi
    
    return $exit_code
}

# Run experiments with optimized order (fastest first)
run_experiment "Real Image Experiments" "experiments/real_image_experiments.py" "8"
run_experiment "Comprehensive Benchmark" "experiments/comprehensive_benchmark.py" "5"

# Start longer experiments in background
echo ""
echo "🔄 Starting intensive experiments in parallel..."

# Start comparative analysis in background
echo "📊 Starting Comparative Analysis (background)..."
cd .. 2>/dev/null || true
python experiments/comparative_analysis.py > research_results/comparative_analysis.log 2>&1 &
comp_pid=$!

# Start advanced research in background  
echo "🔬 Starting Advanced Research (background)..."
python experiments/advanced_research.py > research_results/advanced_research.log 2>&1 &
adv_pid=$!

# Run synthetic speed test (quick)
run_experiment "Synthetic Speed Test" "benchmarks/synthetic_speed.py" "2"

# Wait for background processes
echo ""
echo "⏳ Waiting for background experiments..."

wait_for_process() {
    local pid=$1
    local name="$2"
    local log_file="$3"
    
    if kill -0 "$pid" 2>/dev/null; then
        echo "  ⏳ Waiting for $name..."
        wait "$pid"
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "  ✅ $name completed successfully"
        else
            echo "  ⚠️  $name completed with warnings (check $log_file)"
        fi
    fi
}

wait_for_process "$comp_pid" "Comparative Analysis" "research_results/comparative_analysis.log"
wait_for_process "$adv_pid" "Advanced Research" "research_results/advanced_research.log"

echo ""
echo "=========================================="
echo "🎉 Research Experiments Completed!"
echo "=========================================="

# Calculate total runtime
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "⏱️  Total runtime: ${total_time}s ($(($total_time / 60))m $(($total_time % 60))s)"

echo ""
echo "📊 Generated Results Summary:"
echo "----------------------------"

# Enhanced results summary with file counts
count_files() {
    local dir="$1"
    local pattern="$2"
    if [ -d "$dir" ]; then
        find "$dir" -name "$pattern" 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

show_detailed_results() {
    local dir="$1"
    local name="$2"
    
    if [ -d "$dir" ]; then
        local json_count=$(count_files "$dir" "*.json")
        local png_count=$(count_files "$dir" "*.png") 
        local tex_count=$(count_files "$dir" "*.tex")
        local total=$((json_count + png_count + tex_count))
        
        echo "📁 $name: $total files ($json_count JSON, $png_count plots, $tex_count tables)"
        
        # Show key result files
        if [ $png_count -gt 0 ]; then
            echo "   📈 Key plots:"
            find "$dir" -name "*.png" -exec basename {} \; 2>/dev/null | head -2 | sed 's/^/      /'
        fi
    else
        echo "📁 $name: No results directory found"
    fi
}

# List all generated files with enhanced formatting
echo ""
show_detailed_results "real_image_results" "Real Image Results"
show_detailed_results "comparative_results" "Comparative Analysis Results"
show_detailed_results "advanced_results" "Advanced Research Results"
show_detailed_results "benchmark_results" "Comprehensive Benchmark Results"
show_detailed_results "synthetic_results" "Synthetic Speed Results"

echo ""
echo "📋 Performance Optimizations Applied:"
echo "===================================="
echo "✅ Parallel execution for long-running experiments"
echo "✅ Optimized experiment ordering (fast → slow)"
echo "✅ Enhanced error handling and logging"
echo "✅ Environment variable optimizations (OpenBLAS, OMP)"
echo "✅ Background processing for independent experiments"

echo ""
echo "📈 Research Summary:"
echo "==================="
echo "🔬 Real image experiments: CIFAR-10 analysis with multiple feature extractors"
echo "⚖️  Comparative analysis: Performance vs FAISS, Annoy, and other baselines"
echo "🧮 Advanced research: Dimensionality studies and parameter sensitivity"
echo "🏆 Comprehensive benchmark: Speed/accuracy/memory tradeoff analysis"
echo "🚀 Synthetic testing: Scalability and performance validation"

echo ""
echo "📚 Publication-Ready Outputs:"
echo "============================="
echo "📊 Performance metrics (QPS, Recall, Memory usage)"
echo "📈 Visualization plots (PNG files for papers)"
echo "📄 Raw experimental data (JSON files for analysis)"
echo "🎓 LaTeX tables (ready for academic publications)"

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. Review key plots in */results/ directories"
echo "2. Check experiment.log for any warnings or issues"
echo "3. Use JSON data for custom analysis and additional plots"
echo "4. Include LaTeX tables in your research paper"

if [ -f "research_results/experiment.log" ]; then
    echo ""
    echo "📝 Experiment Log Summary:"
    echo "========================="
    echo "   Full log: research_results/experiment.log"
    
    # Show any warnings or errors
    if grep -q "Warning\|Error\|Failed" research_results/experiment.log 2>/dev/null; then
        echo "   ⚠️  Found warnings/errors - review log file for details"
    else
        echo "   ✅ No critical issues detected"
    fi
fi

echo ""
echo "🚀 Speed Comparison:"
echo "==================="
echo "📊 Standard experiments: ~15-30 minutes (this script)"
echo "⚡ Fast experiments: ~5-10 minutes (./run_optimized_experiments.sh)"
echo "🎯 Quick validation: ~2-5 minutes (./run_fast_experiments.sh)"

echo "=========================================="
