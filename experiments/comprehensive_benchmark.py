"""
Comprehensive benchmarking framework comparing Asteria with existing methods
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
from collections import defaultdict
import argparse

# Asteria imports
from asteria.bor import ButterflyRotation
from asteria.ecvh import ECVH
from asteria.lrsq import LRSQ
from asteria.index_cpu import AsteriaIndexCPU

class BaselineMethod:
    """Base class for baseline methods"""
    def __init__(self, name: str):
        self.name = name
    
    def build_index(self, vectors: np.ndarray):
        raise NotImplementedError
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def get_memory_usage(self) -> float:
        """Return memory usage in MB"""
        return 0.0

class FAISSBaseline(BaselineMethod):
    """FAISS IVF baseline"""
    def __init__(self, nlist: int = 100):
        super().__init__("FAISS-IVF")
        self.nlist = nlist
        self.index = None
        
    def build_index(self, vectors: np.ndarray):
        try:
            import faiss
            d = vectors.shape[1]
            n_vectors = vectors.shape[0]
            
            # Adjust nlist based on dataset size
            # FAISS needs at least 39*nlist training points
            max_nlist = max(1, n_vectors // 100)
            actual_nlist = min(self.nlist, max_nlist)
            
            if n_vectors < 1000:
                # Use flat index for small datasets
                self.index = faiss.IndexFlatIP(d)
                self.name = "FAISS-Flat"
            else:
                quantizer = faiss.IndexFlatIP(d)
                self.index = faiss.IndexIVFFlat(quantizer, d, actual_nlist)
                self.index.train(vectors.astype('float32'))
                self.name = f"FAISS-IVF{actual_nlist}"
            
            self.index.add(vectors.astype('float32'))
        except ImportError:
            print("FAISS not available. Install with: pip install faiss-cpu")
            self.index = None
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            return np.zeros((queries.shape[0], k)), np.zeros((queries.shape[0], k))
        
        distances, indices = self.index.search(queries.astype('float32'), k)
        return distances, indices
    
    def get_memory_usage(self) -> float:
        if self.index is None:
            return 0.0
        # Estimate memory usage
        return self.index.ntotal * self.index.d * 4 / (1024 * 1024)  # 4 bytes per float

class AnnoyBaseline(BaselineMethod):
    """Annoy baseline"""
    def __init__(self, n_trees: int = 10):
        super().__init__("Annoy")
        self.n_trees = n_trees
        self.index = None
        self.vectors = None
        
    def build_index(self, vectors: np.ndarray):
        try:
            from annoy import AnnoyIndex
            f = vectors.shape[1]
            self.index = AnnoyIndex(f, 'angular')
            self.vectors = vectors
            
            for i, v in enumerate(vectors):
                self.index.add_item(i, v)
            self.index.build(self.n_trees)
        except ImportError:
            print("Annoy not available. Install with: pip install annoy")
            self.index = None
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            return np.zeros((queries.shape[0], k)), np.zeros((queries.shape[0], k))
        
        indices = []
        distances = []
        
        for query in queries:
            ids, dists = self.index.get_nns_by_vector(query, k, include_distances=True)
            indices.append(ids)
            distances.append(dists)
        
        return np.array(distances), np.array(indices)

class BruteForceBaseline(BaselineMethod):
    """Brute force exact search baseline"""
    def __init__(self):
        super().__init__("BruteForce")
        self.vectors = None
        
    def build_index(self, vectors: np.ndarray):
        self.vectors = vectors
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectors is None:
            return np.zeros((queries.shape[0], k)), np.zeros((queries.shape[0], k))
        
        # Compute all pairwise distances
        similarities = np.dot(queries, self.vectors.T)
        indices = np.argpartition(-similarities, k-1, axis=1)[:, :k]
        
        # Sort the top-k
        rows = np.arange(queries.shape[0])[:, np.newaxis]
        top_similarities = similarities[rows, indices]
        sort_indices = np.argsort(-top_similarities, axis=1)
        
        final_indices = indices[rows, sort_indices]
        final_distances = 1 - top_similarities[rows, sort_indices]  # Convert to distance
        
        return final_distances, final_indices

class AsteriaMethod(BaselineMethod):
    """Asteria method wrapper"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Asteria")
        self.config = config
        self.index = None
        self.model_bundle = None
        
    def build_index(self, vectors: np.ndarray):
        dim = vectors.shape[1]
        
        # Create models
        bor = ButterflyRotation(dim)
        ecvh = ECVH(dim, 
                   self.config.get('m_vantages', 48),
                   self.config.get('raw_bits', 32), 
                   self.config.get('code_bits', 32))
        lrsq = LRSQ(dim, 
                   self.config.get('rank', 48),
                   self.config.get('blocks', 12))
        
        self.model_bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
        self.index = AsteriaIndexCPU(self.model_bundle, device='cpu')
        
        # Add vectors to index
        vectors_tensor = torch.tensor(vectors, dtype=torch.float32)
        self.index.add(vectors_tensor)
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            return np.zeros((queries.shape[0], k)), np.zeros((queries.shape[0], k))
        
        queries_tensor = torch.tensor(queries, dtype=torch.float32)
        distances, indices = self.index.search(
            queries_tensor, 
            k=k,
            target_mult=self.config.get('target_mult', 8),
            max_radius=self.config.get('max_radius', 2)
        )
        
        # Convert to numpy if they are tensors
        if isinstance(distances, torch.Tensor):
            distances = distances.numpy()
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
            
        return distances, indices

class ComprehensiveBenchmark:
    """Comprehensive benchmarking framework"""
    
    def __init__(self, save_dir: str = "benchmark_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results = defaultdict(list)
        
    def run_scale_experiment(self, 
                           db_sizes: List[int] = [1000, 5000, 10000, 50000, 100000],
                           query_size: int = 1000,
                           dim: int = 768,
                           k: int = 10):
        """Experiment: Performance vs database size"""
        
        print("=== Running Scale Experiment ===")
        
        methods = {
            'asteria': AsteriaMethod({
                'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48,
                'rank': 48, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
            }),
            'brute_force': BruteForceBaseline(),
            'faiss': FAISSBaseline(nlist=min(100, max(db_sizes) // 100)),
            'annoy': AnnoyBaseline(n_trees=10)
        }
        
        for db_size in db_sizes:
            print(f"\nTesting with database size: {db_size}")
            
            # Generate synthetic data
            db_vectors = np.random.randn(db_size, dim).astype('float32')
            db_vectors = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
            
            query_vectors = np.random.randn(query_size, dim).astype('float32')
            query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
            
            # Get ground truth from brute force
            gt_method = BruteForceBaseline()
            gt_method.build_index(db_vectors)
            gt_distances, gt_indices = gt_method.search(query_vectors, k)
            
            for method_name, method in methods.items():
                print(f"  Testing {method_name}...")
                
                # Build index
                build_start = time.time()
                method.build_index(db_vectors)
                build_time = time.time() - build_start
                
                # Search
                search_start = time.time()
                distances, indices = method.search(query_vectors, k)
                search_time = time.time() - search_start
                
                # Calculate metrics
                qps = query_size / search_time if search_time > 0 else 0
                memory_mb = method.get_memory_usage()
                
                # Calculate recall
                recall = self._calculate_recall(indices, gt_indices)
                
                self.results['scale_experiment'].append({
                    'method': method_name,
                    'db_size': db_size,
                    'build_time': build_time,
                    'search_time': search_time,
                    'qps': qps,
                    'memory_mb': memory_mb,
                    'recall@10': recall,
                    'dim': dim,
                    'k': k
                })
        
        self._plot_scale_results()
    
    def run_parameter_study(self, 
                          base_config: Dict[str, Any],
                          param_ranges: Dict[str, List[Any]],
                          db_size: int = 10000,
                          query_size: int = 1000,
                          dim: int = 768):
        """Study effect of different parameters on Asteria performance"""
        
        print("=== Running Parameter Study ===")
        
        # Generate test data
        db_vectors = np.random.randn(db_size, dim).astype('float32')
        db_vectors = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
        
        query_vectors = np.random.randn(query_size, dim).astype('float32')
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        
        # Ground truth
        gt_method = BruteForceBaseline()
        gt_method.build_index(db_vectors)
        gt_distances, gt_indices = gt_method.search(query_vectors, 10)
        
        for param_name, param_values in param_ranges.items():
            print(f"\nStudying parameter: {param_name}")
            
            for param_value in param_values:
                config = base_config.copy()
                config[param_name] = param_value
                
                # Validate ECVH parameters
                if 'code_bits' in config and 'raw_bits' in config:
                    if config['code_bits'] < config['raw_bits']:
                        config['code_bits'] = config['raw_bits']
                
                print(f"  Testing {param_name}={param_value}")
                
                try:
                    method = AsteriaMethod(config)
                    
                    # Build and search
                    build_start = time.time()
                    method.build_index(db_vectors)
                    build_time = time.time() - build_start
                    
                    search_start = time.time()
                    distances, indices = method.search(query_vectors, 10)
                    search_time = time.time() - search_start
                    
                    qps = query_size / search_time if search_time > 0 else 0
                    recall = self._calculate_recall(indices, gt_indices)
                    
                    self.results['parameter_study'].append({
                        'parameter': param_name,
                        'value': param_value,
                        'build_time': build_time,
                        'search_time': search_time,
                        'qps': qps,
                        'recall@10': recall,
                        'config': config.copy()
                    })
                
                except Exception as e:
                    print(f"    Error with {param_name}={param_value}: {e}")
        
        self._plot_parameter_results()
    
    def run_memory_vs_accuracy(self, 
                              db_size: int = 50000,
                              query_size: int = 1000,
                              dim: int = 768):
        """Study memory vs accuracy tradeoffs"""
        
        print("=== Running Memory vs Accuracy Study ===")
        
        # Different configurations with varying memory requirements
        configs = [
            {'raw_bits': 16, 'code_bits': 16, 'm_vantages': 32, 'rank': 32, 'blocks': 8},
            {'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48, 'rank': 48, 'blocks': 12},
            {'raw_bits': 64, 'code_bits': 64, 'm_vantages': 64, 'rank': 64, 'blocks': 16},
            {'raw_bits': 128, 'code_bits': 128, 'm_vantages': 128, 'rank': 96, 'blocks': 24},
        ]
        
        # Generate test data
        db_vectors = np.random.randn(db_size, dim).astype('float32')
        db_vectors = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
        
        query_vectors = np.random.randn(query_size, dim).astype('float32')
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        
        # Ground truth
        gt_method = BruteForceBaseline()
        gt_method.build_index(db_vectors)
        gt_distances, gt_indices = gt_method.search(query_vectors, 10)
        
        for i, config in enumerate(configs):
            print(f"Testing config {i+1}/{len(configs)}: {config}")
            
            method = AsteriaMethod(config)
            
            build_start = time.time()
            method.build_index(db_vectors)
            build_time = time.time() - build_start
            
            search_start = time.time()
            distances, indices = method.search(query_vectors, 10)
            search_time = time.time() - search_start
            
            qps = query_size / search_time if search_time > 0 else 0
            recall = self._calculate_recall(indices, gt_indices)
            
            # Estimate memory usage (simplified)
            memory_estimate = (config['code_bits'] * db_size / 8 + 
                             config['m_vantages'] * dim * 4 + 
                             config['rank'] * dim * 4) / (1024 * 1024)
            
            self.results['memory_accuracy'].append({
                'config_id': i,
                'raw_bits': config['raw_bits'],
                'code_bits': config['code_bits'],
                'build_time': build_time,
                'search_time': search_time,
                'qps': qps,
                'recall@10': recall,
                'memory_mb': memory_estimate,
                'config': config.copy()
            })
        
        self._plot_memory_accuracy_results()
    
    def _calculate_recall(self, retrieved_indices: np.ndarray, ground_truth_indices: np.ndarray) -> float:
        """Calculate recall@k"""
        recall_sum = 0
        for i in range(len(retrieved_indices)):
            retrieved_set = set(retrieved_indices[i])
            gt_set = set(ground_truth_indices[i])
            recall_sum += len(retrieved_set & gt_set) / len(gt_set)
        
        return recall_sum / len(retrieved_indices)
    
    def _plot_scale_results(self):
        """Plot scale experiment results"""
        if 'scale_experiment' not in self.results:
            return
        
        try:
            import pandas as pd
            df = pd.DataFrame(self.results['scale_experiment'])
        except ImportError:
            print("Pandas not available, using basic plotting...")
            # Create simple plots without pandas
            results = self.results['scale_experiment']
            methods = list(set([r['method'] for r in results]))
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            for method in methods:
                method_data = [r for r in results if r['method'] == method]
                db_sizes = [r['db_size'] for r in method_data]
                qps_vals = [r['qps'] for r in method_data]
                recall_vals = [r['recall@10'] for r in method_data]
                build_times = [r['build_time'] for r in method_data]
                
                # QPS plot
                axes[0, 0].plot(db_sizes, qps_vals, 'o-', label=method, linewidth=2)
                axes[0, 1].plot(db_sizes, recall_vals, 'o-', label=method, linewidth=2)
                axes[1, 0].plot(db_sizes, build_times, 'o-', label=method, linewidth=2)
            
            axes[0, 0].set_xlabel('Database Size')
            axes[0, 0].set_ylabel('QPS')
            axes[0, 0].set_title('Search Speed vs Database Size')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            axes[0, 0].set_xscale('log')
            axes[0, 0].set_yscale('log')
            
            axes[0, 1].set_xlabel('Database Size')
            axes[0, 1].set_ylabel('Recall@10')
            axes[0, 1].set_title('Recall vs Database Size')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            axes[0, 1].set_xscale('log')
            
            axes[1, 0].set_xlabel('Database Size')
            axes[1, 0].set_ylabel('Build Time (seconds)')
            axes[1, 0].set_title('Build Time vs Database Size')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_yscale('log')
            
            # Hide the unused subplot
            axes[1, 1].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/scale_experiment.png', dpi=300, bbox_inches='tight')
            plt.show()
            return
        
        df = pd.DataFrame(self.results['scale_experiment'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # QPS vs Database Size
        ax1 = axes[0, 0]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            ax1.plot(method_data['db_size'], method_data['qps'], 'o-', label=method, linewidth=2)
        ax1.set_xlabel('Database Size')
        ax1.set_ylabel('Queries Per Second (QPS)')
        ax1.set_title('Search Speed vs Database Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Recall vs Database Size
        ax2 = axes[0, 1]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            ax2.plot(method_data['db_size'], method_data['recall@10'], 'o-', label=method, linewidth=2)
        ax2.set_xlabel('Database Size')
        ax2.set_ylabel('Recall@10')
        ax2.set_title('Recall vs Database Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Build Time vs Database Size
        ax3 = axes[1, 0]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            ax3.plot(method_data['db_size'], method_data['build_time'], 'o-', label=method, linewidth=2)
        ax3.set_xlabel('Database Size')
        ax3.set_ylabel('Build Time (seconds)')
        ax3.set_title('Index Build Time vs Database Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # Memory Usage vs Database Size
        ax4 = axes[1, 1]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            if 'memory_mb' in method_data.columns:
                ax4.plot(method_data['db_size'], method_data['memory_mb'], 'o-', label=method, linewidth=2)
        ax4.set_xlabel('Database Size')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Memory Usage vs Database Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/scale_experiment.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_parameter_results(self):
        """Plot parameter study results"""
        if 'parameter_study' not in self.results:
            return
        
        import pandas as pd
        df = pd.DataFrame(self.results['parameter_study'])
        
        parameters = df['parameter'].unique()
        n_params = len(parameters)
        
        fig, axes = plt.subplots(n_params, 2, figsize=(12, 4*n_params))
        if n_params == 1:
            axes = axes.reshape(1, -1)
        
        for i, param in enumerate(parameters):
            param_data = df[df['parameter'] == param]
            
            # QPS vs Parameter
            ax1 = axes[i, 0]
            ax1.plot(param_data['value'], param_data['qps'], 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel(f'{param}')
            ax1.set_ylabel('QPS')
            ax1.set_title(f'QPS vs {param}')
            ax1.grid(True, alpha=0.3)
            
            # Recall vs Parameter
            ax2 = axes[i, 1]
            ax2.plot(param_data['value'], param_data['recall@10'], 'o-', linewidth=2, markersize=8, color='red')
            ax2.set_xlabel(f'{param}')
            ax2.set_ylabel('Recall@10')
            ax2.set_title(f'Recall@10 vs {param}')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/parameter_study.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_memory_accuracy_results(self):
        """Plot memory vs accuracy tradeoff"""
        if 'memory_accuracy' not in self.results:
            return
        
        import pandas as pd
        df = pd.DataFrame(self.results['memory_accuracy'])
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Memory vs Recall
        ax1 = axes[0]
        scatter = ax1.scatter(df['memory_mb'], df['recall@10'], 
                            c=df['code_bits'], cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('Memory Usage (MB)')
        ax1.set_ylabel('Recall@10')
        ax1.set_title('Memory vs Accuracy Tradeoff')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar for code_bits
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Code Bits')
        
        # Add annotations
        for i, row in df.iterrows():
            ax1.annotate(f"{row['code_bits']}bits", 
                        (row['memory_mb'], row['recall@10']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # QPS vs Memory
        ax2 = axes[1]
        scatter2 = ax2.scatter(df['memory_mb'], df['qps'], 
                             c=df['code_bits'], cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Memory Usage (MB)')
        ax2.set_ylabel('QPS')
        ax2.set_title('Memory vs Speed Tradeoff')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Code Bits')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/memory_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str = 'benchmark_results.json'):
        """Save all results to JSON file"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            json.dump(dict(self.results), f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def generate_summary_report(self):
        """Generate a summary report of all experiments"""
        report_path = os.path.join(self.save_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("ASTERIA BENCHMARK SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for experiment_name, experiment_data in self.results.items():
                f.write(f"{experiment_name.upper()}\n")
                f.write("-" * 30 + "\n")
                
                if experiment_name == 'scale_experiment':
                    asteria_data = [d for d in experiment_data if d['method'] == 'Asteria']
                    if asteria_data:
                        best_qps = max(asteria_data, key=lambda x: x['qps'])
                        f.write(f"Best QPS: {best_qps['qps']:.2f} at DB size {best_qps['db_size']}\n")
                        
                        avg_recall = np.mean([d['recall@10'] for d in asteria_data])
                        f.write(f"Average Recall@10: {avg_recall:.4f}\n")
                
                elif experiment_name == 'memory_accuracy':
                    best_tradeoff = max(experiment_data, key=lambda x: x['recall@10'] / x['memory_mb'])
                    f.write(f"Best Memory/Accuracy Tradeoff: {best_tradeoff['recall@10']:.4f} recall ")
                    f.write(f"at {best_tradeoff['memory_mb']:.2f} MB\n")
                
                f.write("\n")
        
        print(f"Summary report generated: {report_path}")

def main():
    """Run comprehensive benchmarks"""
    
    parser = argparse.ArgumentParser(description='Comprehensive Benchmark for Asteria')
    parser.add_argument('--fast-mode', action='store_true', 
                       help='Run in fast mode with reduced dataset sizes')
    parser.add_argument('--output-dir', type=str, default='research_results_optimized/benchmark_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set environment variables based on arguments
    if args.fast_mode:
        os.environ['ASTERIA_FAST_MODE'] = '1'
        os.environ['ASTERIA_SMALL_DATASETS'] = '1'
    
    benchmark = ComprehensiveBenchmark(args.output_dir)
    
    try:
        # Adjust parameters based on fast mode
        if args.fast_mode or os.getenv('ASTERIA_FAST_MODE', '0') == '1':
            db_sizes = [1000, 2500, 5000]
            query_size = 200
            dim = 512
            print("🚀 Running in FAST MODE - using reduced dataset sizes")
        else:
            db_sizes = [1000, 5000, 10000, 25000, 50000]
            query_size = 1000
            dim = 768
        
        # 1. Scale experiment
        print("Running scale experiment...")
        benchmark.run_scale_experiment(
            db_sizes=db_sizes,
            query_size=query_size,
            dim=dim
        )
        
        # 2. Parameter study
        print("Running parameter study...")
        base_config = {
            'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48,
            'rank': 48, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
        }
        
        param_ranges = {
            'code_bits': [16, 32, 64],  # Reduced for fast mode
            'm_vantages': [24, 48, 96],
            'max_radius': [1, 2, 3]
        }
        
        benchmark.run_parameter_study(base_config, param_ranges, 
                                    db_size=db_sizes[-1], query_size=query_size)
        
        # 3. Memory vs accuracy study
        print("Running memory vs accuracy study...")
        benchmark.run_memory_vs_accuracy(db_size=db_sizes[-1], query_size=query_size)
        
        # Save results and generate report
        benchmark.save_results()
        benchmark.generate_summary_report()
        
        print(f"All benchmarks completed! Check the '{benchmark.save_dir}' directory for plots and data.")
    except Exception as e:
        print(f"Error in comprehensive benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
