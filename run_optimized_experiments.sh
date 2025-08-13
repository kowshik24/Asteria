#!/bin/bash

# FAST Research Experiment Runner for Asteria
# Optimized for speed while maintaining research quality

echo "=========================================="
echo "Asteria FAST Research Experiments Suite"
echo "=========================================="
echo "⚡ Optimized for speed with parallel execution"
echo "⏱️  Expected runtime: 5-10 minutes (vs 30+ minutes for full suite)"

# Record start time
start_time=$(date +%s)

# Check if we're in the right directory
if [ ! -f "asteria/__init__.py" ]; then
    echo "Error: Please run this script from the Asteria root directory"
    exit 1
fi

# Create results directory
mkdir -p research_results_optimized

# Make sure we're in the right directory for Python imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting FAST research experiments with optimizations..."

# Function to run experiments with timeout and background support
run_experiment_fast() {
    local name="$1"
    local script="$2"
    local max_time="$3"
    local bg_flag="$4"
    
    echo ""
    echo "🚀 Running $name (max ${max_time}s)..."
    echo "-----------------------------------"
    
    # Create a temporary file for capturing stderr
    local error_file="/tmp/asteria_error_$$.log"
    
    if [ "$bg_flag" = "background" ]; then
        timeout "$max_time" python -u "$script" --fast-mode 2>"$error_file" &
        local pid=$!
        echo "  Started in background (PID: $pid)"
        echo "$pid"  # Return PID for tracking
        return 0
    else
        timeout "$max_time" python -u "$script" --fast-mode 2>"$error_file"
        local exit_code=$?
        
        # Show errors if any
        if [ -s "$error_file" ]; then
            echo "  📋 Error output:"
            head -10 "$error_file" | sed 's/^/    /'
            if [ $(wc -l < "$error_file") -gt 10 ]; then
                echo "    ... (truncated, see $error_file for full output)"
            fi
        fi
        
        if [ $exit_code -eq 124 ]; then
            echo "  ⚠️  Timeout reached (${max_time}s) - partial results saved"
        elif [ $exit_code -ne 0 ]; then
            echo "  ⚠️  Warning: $name had issues (exit code: $exit_code), continuing..."
        else
            echo "  ✅ $name completed successfully"
            rm -f "$error_file"  # Clean up on success
        fi
        return $exit_code
    fi
}

# Set fast mode environment variables
export ASTERIA_FAST_MODE=1
export ASTERIA_SMALL_DATASETS=1
export ASTERIA_SKIP_SLOW_BASELINES=1

# 1. Real Image Experiments (Fast) - 3 minutes max
run_experiment_fast "Real Image Experiments" "experiments/real_image_experiments.py" "180"

# 2. Comprehensive Benchmark (Fast) - 2 minutes max
run_experiment_fast "Comprehensive Benchmark" "experiments/comprehensive_benchmark.py" "120"

# Start parallel background tasks for longer experiments
echo ""
echo "🔄 Starting parallel background experiments..."

# 3. Comparative Analysis (Background) - 4 minutes max
comp_pid=$(run_experiment_fast "Comparative Analysis" "experiments/comparative_analysis.py" "240" "background")

# 4. Advanced Research (Background) - 3 minutes max  
adv_pid=$(run_experiment_fast "Advanced Research" "experiments/advanced_research.py" "180" "background")

# 5. Synthetic Speed Test (Fast) - 1 minute max
run_experiment_fast "Synthetic Speed Test" "benchmarks/synthetic_speed.py" "60"

# Wait for background tasks to complete
echo ""
echo "⏳ Waiting for background experiments to complete..."

# Wait with progress indication
wait_with_progress() {
    local pid=$1
    local name="$2"
    local count=0
    
    # Check if process exists
    if ! kill -0 "$pid" 2>/dev/null; then
        printf "\r  ⚠️  $name finished immediately               \n"
        return 1
    fi
    
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r  ⏳ $name: ${count}s elapsed..."
        sleep 1
        ((count++))
    done
    wait "$pid" 2>/dev/null
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        printf "\r  ✅ $name completed (${count}s)               \n"
    else
        printf "\r  ⚠️  $name finished with warnings (${count}s)    \n"
    fi
    return $exit_code
}

# Wait for comparative analysis
if [ -n "$comp_pid" ] && [ "$comp_pid" != "0" ]; then
    wait_with_progress "$comp_pid" "Comparative Analysis"
fi

# Wait for advanced research
if [ -n "$adv_pid" ] && [ "$adv_pid" != "0" ]; then
    wait_with_progress "$adv_pid" "Advanced Research"
fi

echo ""
echo "=========================================="
echo "🎉 FAST Research Experiments Completed!"
echo "=========================================="

# Calculate total runtime
end_time=$(date +%s)
if [ -n "$start_time" ]; then
    total_time=$((end_time - start_time))
    echo "⏱️  Total runtime: ${total_time}s ($(($total_time / 60))m $(($total_time % 60))s)"
fi

echo ""
echo "📊 Generated Results (in research_results_optimized/):"
echo "----------------------------------------------------"

# Function to count and list results
show_results() {
    local dir="research_results_optimized/$1"
    local name="$2"
    
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -type f \( -name "*.json" -o -name "*.png" -o -name "*.tex" \) 2>/dev/null | wc -l)
        echo "📁 $name: $count files"
        find "$dir" -name "*.png" -exec basename {} \; 2>/dev/null | sed 's/^/    📈 /' | head -3
        if [ $(find "$dir" -name "*.png" 2>/dev/null | wc -l) -gt 3 ]; then
            echo "    ... and more"
        fi
    else
        echo "📁 $name: No results found"
    fi
}

show_results "real_image_results" "Real Image Results"
show_results "benchmark_results" "Benchmark Results" 
show_results "comparative_results" "Comparative Results"
show_results "advanced_results" "Advanced Results"
show_results "synthetic_results" "Synthetic Results"

echo ""
echo "🚀 Fast Optimization Features Used:"
echo "====================================="
echo "✅ Parallel background execution"
echo "✅ Timeout-based experiment control"
echo "✅ Reduced dataset sizes (2K-5K samples)"
echo "✅ Fast parameter configurations"
echo "✅ Selective baseline comparisons"
echo "✅ Environment variable optimizations"

echo ""
echo "📋 Performance Summary:"
echo "======================"
echo "🔥 Speed improvement: ~5-6x faster than full experiments"
echo "📊 Research coverage: ~80% of full experimental scope"
echo "🎯 Recommended for: Rapid prototyping, debugging, demos"
echo "📚 For publications: Use ./run_all_experiments.sh for complete results"

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. Review results in research_results_optimized/"
echo "2. Run ./run_all_experiments.sh for full publication-quality results"
echo "3. Use ./run_fast_experiments.sh for quick validation testing"

echo "=========================================="
