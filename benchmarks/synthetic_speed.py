"""
Enhanced synthetic speed benchmark with comprehensive analysis and plotting
"""
import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

from asteria.bor import ButterflyRotation
from asteria.ecvh import ECVH
from asteria.lrsq import LRSQ
from asteria.index_cpu import AsteriaIndexCPU

# Set style for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

class SyntheticBenchmark:
    """Enhanced synthetic benchmark with comprehensive analysis"""
    
    def __init__(self, save_dir: str = "synthetic_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results = {}
    
    def run_single_test(self, 
                       dim: int = 768,
                       train_size: int = 20000,
                       db_size: int = 100000,
                       query_size: int = 1000,
                       config: Dict = None) -> Dict:
        """Run a single benchmark test"""
        
        if config is None:
            config = {
                'm_vantages': 160,
                'raw_bits': 96,
                'code_bits': 128,
                'rank': 96,
                'blocks': 24,
                'target_mult': 8,
                'max_radius': 2
            }
        
        print(f"Running test: dim={dim}, db_size={db_size}, query_size={query_size}")
        print(f"Config: {config}")
        
        # Generate training data
        print("Generating training data...")
        train = torch.randn(train_size, dim)
        train = train / train.norm(dim=1, keepdim=True)
        
        # Create models
        print("Creating models...")
        bor = ButterflyRotation(dim)
        ecvh = ECVH(dim, config['m_vantages'], config['raw_bits'], config['code_bits'])
        lrsq = LRSQ(dim, config['rank'], config['blocks'])
        bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
        
        # Create index
        index = AsteriaIndexCPU(bundle)
        
        # Generate database
        print("Generating database...")
        db = torch.randn(db_size, dim)
        db = db / db.norm(dim=1, keepdim=True)
        
        # Add to index
        print("Building index...")
        build_start = time.time()
        index.add(db)
        build_time = time.time() - build_start
        
        # Generate queries
        print("Generating queries...")
        queries = torch.randn(query_size, dim)
        queries = queries / queries.norm(dim=1, keepdim=True)
        
        # Search benchmark
        print("Running search benchmark...")
        search_times = []
        for i in range(5):  # Multiple runs for stability
            start_time = time.time()
            D, I = index.search(queries, 
                              k=10, 
                              target_mult=config['target_mult'],
                              max_radius=config['max_radius'])
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        avg_search_time = np.mean(search_times)
        qps = query_size / avg_search_time
        
        # Calculate recall (approximate)
        print("Calculating recall...")
        gt_similarities = torch.mm(queries, db.T)
        _, gt_indices = torch.topk(gt_similarities, k=10, dim=1)
        
        # Calculate recall@10
        recall_sum = 0
        for i in range(query_size):
            retrieved_set = set(I[i].tolist())
            gt_set = set(gt_indices[i].tolist())
            recall_sum += len(retrieved_set & gt_set) / 10
        
        recall_at_10 = recall_sum / query_size
        
        results = {
            'dim': dim,
            'train_size': train_size,
            'db_size': db_size,
            'query_size': query_size,
            'config': config,
            'build_time': build_time,
            'search_times': search_times,
            'avg_search_time': avg_search_time,
            'qps': qps,
            'recall_at_10': recall_at_10,
            'memory_estimate_mb': self._estimate_memory(config, db_size, dim)
        }
        
        print(f"Results: QPS={qps:.2f}, Recall@10={recall_at_10:.4f}, Build_time={build_time:.2f}s")
        return results
    
    def run_scale_benchmark(self, 
                          db_sizes: List[int] = [1000, 5000, 10000, 50000, 100000, 200000],
                          dim: int = 768,
                          query_size: int = 1000):
        """Benchmark performance across different database sizes"""
        
        print("=== Running Scale Benchmark ===")
        
        results = []
        for db_size in db_sizes:
            try:
                result = self.run_single_test(
                    dim=dim,
                    db_size=db_size,
                    query_size=query_size,
                    config={
                        'm_vantages': 48,
                        'raw_bits': 32,
                        'code_bits': 32,
                        'rank': 48,
                        'blocks': 12,
                        'target_mult': 8,
                        'max_radius': 2
                    }
                )
                results.append(result)
            except Exception as e:
                print(f"Error with db_size {db_size}: {e}")
                continue
        
        self.results['scale_benchmark'] = results
        self._plot_scale_results(results)
        return results
    
    def run_parameter_sweep(self, 
                          db_size: int = 50000,
                          dim: int = 768,
                          query_size: int = 1000):
        """Sweep different parameter configurations"""
        
        print("=== Running Parameter Sweep ===")
        
        parameter_configs = [
            # Small/Fast configs
            {'m_vantages': 24, 'raw_bits': 16, 'code_bits': 16, 'rank': 24, 'blocks': 8, 'name': 'Small'},
            {'m_vantages': 32, 'raw_bits': 24, 'code_bits': 24, 'rank': 32, 'blocks': 8, 'name': 'Medium-Small'},
            # Medium configs
            {'m_vantages': 48, 'raw_bits': 32, 'code_bits': 32, 'rank': 48, 'blocks': 12, 'name': 'Medium'},
            {'m_vantages': 64, 'raw_bits': 48, 'code_bits': 48, 'rank': 64, 'blocks': 16, 'name': 'Medium-Large'},
            # Large/Accurate configs
            {'m_vantages': 96, 'raw_bits': 64, 'code_bits': 64, 'rank': 96, 'blocks': 24, 'name': 'Large'},
            {'m_vantages': 128, 'raw_bits': 96, 'code_bits': 96, 'rank': 128, 'blocks': 32, 'name': 'Extra-Large'},
        ]
        
        results = []
        for config in parameter_configs:
            try:
                config_copy = config.copy()
                name = config_copy.pop('name')
                config_copy.update({'target_mult': 8, 'max_radius': 2})
                
                result = self.run_single_test(
                    dim=dim,
                    db_size=db_size,
                    query_size=query_size,
                    config=config_copy
                )
                result['config_name'] = name
                results.append(result)
            except Exception as e:
                print(f"Error with config {config}: {e}")
                continue
        
        self.results['parameter_sweep'] = results
        self._plot_parameter_sweep_results(results)
        return results
    
    def run_dimension_study(self, 
                          dimensions: List[int] = [128, 256, 512, 768, 1024, 1536],
                          db_size: int = 20000,
                          query_size: int = 1000):
        """Study performance across different vector dimensions"""
        
        print("=== Running Dimension Study ===")
        
        results = []
        for dim in dimensions:
            try:
                result = self.run_single_test(
                    dim=dim,
                    db_size=db_size,
                    query_size=query_size,
                    config={
                        'm_vantages': min(48, dim // 16),  # Adaptive based on dimension
                        'raw_bits': 32,
                        'code_bits': 32,
                        'rank': min(48, dim // 16),
                        'blocks': 12,
                        'target_mult': 8,
                        'max_radius': 2
                    }
                )
                results.append(result)
            except Exception as e:
                print(f"Error with dimension {dim}: {e}")
                continue
        
        self.results['dimension_study'] = results
        self._plot_dimension_results(results)
        return results
    
    def _estimate_memory(self, config: Dict, db_size: int, dim: int) -> float:
        """Estimate memory usage in MB"""
        # Hash codes storage
        hash_memory = db_size * config['code_bits'] / 8
        
        # Vantage points storage
        vantage_memory = config['m_vantages'] * dim * 4  # float32
        
        # LRSQ parameters
        lrsq_memory = config['rank'] * dim * config['blocks'] * 4
        
        # Butterfly rotation parameters (approximate)
        bor_memory = dim * np.log2(dim) * 4
        
        total_bytes = hash_memory + vantage_memory + lrsq_memory + bor_memory
        return total_bytes / (1024 * 1024)
    
    def _plot_scale_results(self, results: List[Dict]):
        """Plot scale benchmark results"""
        
        db_sizes = [r['db_size'] for r in results]
        qps_values = [r['qps'] for r in results]
        recall_values = [r['recall_at_10'] for r in results]
        build_times = [r['build_time'] for r in results]
        memory_values = [r['memory_estimate_mb'] for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # QPS vs Database Size
        axes[0, 0].semilogx(db_sizes, qps_values, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Database Size')
        axes[0, 0].set_ylabel('Queries Per Second (QPS)')
        axes[0, 0].set_title('Search Speed vs Database Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall vs Database Size
        axes[0, 1].semilogx(db_sizes, recall_values, 'o-', linewidth=2, markersize=8, color='red')
        axes[0, 1].set_xlabel('Database Size')
        axes[0, 1].set_ylabel('Recall@10')
        axes[0, 1].set_title('Recall vs Database Size')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.1)
        
        # Build Time vs Database Size
        axes[1, 0].loglog(db_sizes, build_times, 'o-', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel('Database Size')
        axes[1, 0].set_ylabel('Build Time (seconds)')
        axes[1, 0].set_title('Index Build Time vs Database Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Memory vs Database Size
        axes[1, 1].loglog(db_sizes, memory_values, 'o-', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_xlabel('Database Size')
        axes[1, 1].set_ylabel('Memory Usage (MB)')
        axes[1, 1].set_title('Memory Usage vs Database Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/scale_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_parameter_sweep_results(self, results: List[Dict]):
        """Plot parameter sweep results"""
        
        config_names = [r['config_name'] for r in results]
        qps_values = [r['qps'] for r in results]
        recall_values = [r['recall_at_10'] for r in results]
        memory_values = [r['memory_estimate_mb'] for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # QPS by configuration
        bars1 = axes[0, 0].bar(config_names, qps_values, alpha=0.7)
        axes[0, 0].set_xlabel('Configuration')
        axes[0, 0].set_ylabel('QPS')
        axes[0, 0].set_title('Search Speed by Configuration')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, qps in zip(bars1, qps_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(qps_values)*0.01,
                          f'{qps:.1f}', ha='center', va='bottom')
        
        # Recall by configuration
        bars2 = axes[0, 1].bar(config_names, recall_values, alpha=0.7, color='red')
        axes[0, 1].set_xlabel('Configuration')
        axes[0, 1].set_ylabel('Recall@10')
        axes[0, 1].set_title('Recall by Configuration')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, recall in zip(bars2, recall_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{recall:.3f}', ha='center', va='bottom')
        
        # Memory usage by configuration
        bars3 = axes[1, 0].bar(config_names, memory_values, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Configuration')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage by Configuration')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Speed vs Accuracy tradeoff
        scatter = axes[1, 1].scatter(recall_values, qps_values, c=memory_values, 
                                   cmap='viridis', s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Recall@10')
        axes[1, 1].set_ylabel('QPS')
        axes[1, 1].set_title('Speed vs Accuracy Tradeoff')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add configuration labels
        for i, name in enumerate(config_names):
            axes[1, 1].annotate(name, (recall_values[i], qps_values[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/parameter_sweep.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_dimension_results(self, results: List[Dict]):
        """Plot dimension study results"""
        
        dimensions = [r['dim'] for r in results]
        qps_values = [r['qps'] for r in results]
        recall_values = [r['recall_at_10'] for r in results]
        build_times = [r['build_time'] for r in results]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # QPS vs Dimension
        axes[0].plot(dimensions, qps_values, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Vector Dimension')
        axes[0].set_ylabel('QPS')
        axes[0].set_title('Search Speed vs Vector Dimension')
        axes[0].grid(True, alpha=0.3)
        
        # Recall vs Dimension
        axes[1].plot(dimensions, recall_values, 'o-', linewidth=2, markersize=8, color='red')
        axes[1].set_xlabel('Vector Dimension')
        axes[1].set_ylabel('Recall@10')
        axes[1].set_title('Recall vs Vector Dimension')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.1)
        
        # Build Time vs Dimension
        axes[2].plot(dimensions, build_times, 'o-', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Vector Dimension')
        axes[2].set_ylabel('Build Time (seconds)')
        axes[2].set_title('Build Time vs Vector Dimension')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/dimension_study.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str = 'synthetic_benchmark_results.json'):
        """Save all results to JSON"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def generate_summary_table(self):
        """Generate a summary table of all results"""
        print("\n" + "="*80)
        print("ASTERIA SYNTHETIC BENCHMARK SUMMARY")
        print("="*80)
        
        for experiment_name, experiment_results in self.results.items():
            print(f"\n{experiment_name.upper()}:")
            print("-" * 40)
            
            if experiment_name == 'scale_benchmark':
                print(f"{'DB Size':<10} {'QPS':<10} {'Recall@10':<12} {'Build Time':<12} {'Memory (MB)':<12}")
                print("-" * 60)
                for result in experiment_results:
                    print(f"{result['db_size']:<10} {result['qps']:<10.1f} "
                          f"{result['recall_at_10']:<12.4f} {result['build_time']:<12.2f} "
                          f"{result['memory_estimate_mb']:<12.1f}")
            
            elif experiment_name == 'parameter_sweep':
                print(f"{'Config':<15} {'QPS':<10} {'Recall@10':<12} {'Memory (MB)':<12}")
                print("-" * 50)
                for result in experiment_results:
                    print(f"{result['config_name']:<15} {result['qps']:<10.1f} "
                          f"{result['recall_at_10']:<12.4f} {result['memory_estimate_mb']:<12.1f}")
            
            elif experiment_name == 'dimension_study':
                print(f"{'Dimension':<10} {'QPS':<10} {'Recall@10':<12} {'Build Time':<12}")
                print("-" * 45)
                for result in experiment_results:
                    print(f"{result['dim']:<10} {result['qps']:<10.1f} "
                          f"{result['recall_at_10']:<12.4f} {result['build_time']:<12.2f}")

def main():
    """Run comprehensive synthetic benchmarks"""
    
    benchmark = SyntheticBenchmark()
    
    print("Starting comprehensive synthetic benchmarks...")
    
    # 1. Scale benchmark
    print("\n1. Running scale benchmark...")
    benchmark.run_scale_benchmark(
        db_sizes=[1000, 5000, 10000, 25000, 50000, 100000],
        dim=768,
        query_size=1000
    )
    
    # 2. Parameter sweep
    print("\n2. Running parameter sweep...")
    benchmark.run_parameter_sweep(
        db_size=50000,
        dim=768,
        query_size=1000
    )
    
    # 3. Dimension study
    print("\n3. Running dimension study...")
    benchmark.run_dimension_study(
        dimensions=[128, 256, 512, 768, 1024],
        db_size=20000,
        query_size=1000
    )
    
    # Save results and generate summary
    benchmark.save_results()
    benchmark.generate_summary_table()
    
    print(f"\nAll benchmarks completed! Results saved in '{benchmark.save_dir}' directory.")
    print("Generated plots:")
    print("- scale_benchmark.png")
    print("- parameter_sweep.png") 
    print("- dimension_study.png")

if __name__ == "__main__":
    main()