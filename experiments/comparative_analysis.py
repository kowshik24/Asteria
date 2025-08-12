"""
Comparative analysis against state-of-the-art methods
Including FAISS, Annoy, ScaNN, and other baseline methods
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Asteria imports
from asteria.bor import ButterflyRotation
from asteria.ecvh import ECVH
from asteria.lrsq import LRSQ
from asteria.index_cpu import AsteriaIndexCPU

class BaselineMethod:
    """Base class for baseline similarity search methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.index = None
        self.db_vectors = None
        
    def build_index(self, vectors: np.ndarray) -> float:
        """Build index and return build time"""
        raise NotImplementedError
        
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search and return (distances, indices, search_time)"""
        raise NotImplementedError

class FAISSBaseline(BaselineMethod):
    """FAISS baseline implementation"""
    
    def __init__(self, index_type: str = 'IVF'):
        super().__init__(f'FAISS-{index_type}')
        self.index_type = index_type
        
    def build_index(self, vectors: np.ndarray) -> float:
        try:
            import faiss
        except ImportError:
            print("FAISS not available. Skipping FAISS baseline.")
            return 0.0
        
        start_time = time.time()
        
        d = vectors.shape[1]
        
        if self.index_type == 'Flat':
            self.index = faiss.IndexFlatIP(d)
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatIP(d)
            nlist = min(int(np.sqrt(len(vectors))), 1024)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self.index.train(vectors.astype(np.float32))
        elif self.index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(d, 32)
            self.index.hnsw.efConstruction = 200
        elif self.index_type == 'PQ':
            m = min(d // 4, 64)  # number of subquantizers
            self.index = faiss.IndexPQ(d, m, 8)
            self.index.train(vectors.astype(np.float32))
        
        self.index.add(vectors.astype(np.float32))
        return time.time() - start_time
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.index is None:
            return np.array([]), np.array([]), 0.0
        
        if self.index_type == 'IVF':
            self.index.nprobe = 10  # Search 10 clusters
        elif self.index_type == 'HNSW':
            self.index.hnsw.efSearch = 100
        
        start_time = time.time()
        distances, indices = self.index.search(queries.astype(np.float32), k)
        search_time = time.time() - start_time
        
        return distances, indices, search_time

class AnnoyBaseline(BaselineMethod):
    """Annoy baseline implementation"""
    
    def __init__(self, n_trees: int = 100):
        super().__init__(f'Annoy-{n_trees}trees')
        self.n_trees = n_trees
        
    def build_index(self, vectors: np.ndarray) -> float:
        try:
            from annoy import AnnoyIndex
        except ImportError:
            print("Annoy not available. Skipping Annoy baseline.")
            return 0.0
        
        start_time = time.time()
        
        d = vectors.shape[1]
        self.index = AnnoyIndex(d, 'angular')
        
        for i, vector in enumerate(vectors):
            self.index.add_item(i, vector)
        
        self.index.build(self.n_trees)
        return time.time() - start_time
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.index is None:
            return np.array([]), np.array([]), 0.0
        
        start_time = time.time()
        
        indices = []
        distances = []
        
        for query in queries:
            query_indices, query_distances = self.index.get_nns_by_vector(
                query, k, include_distances=True)
            indices.append(query_indices)
            distances.append(query_distances)
        
        search_time = time.time() - start_time
        
        return np.array(distances), np.array(indices), search_time

class BruteForceBaseline(BaselineMethod):
    """Brute force exact search baseline"""
    
    def __init__(self):
        super().__init__('BruteForce')
        
    def build_index(self, vectors: np.ndarray) -> float:
        self.db_vectors = vectors
        return 0.0  # No index building needed
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        start_time = time.time()
        
        similarities = np.dot(queries, self.db_vectors.T)
        indices = np.argpartition(-similarities, k-1, axis=1)[:, :k]
        
        # Get actual distances
        distances = np.array([similarities[i, indices[i]] for i in range(len(queries))])
        
        search_time = time.time() - start_time
        
        return distances, indices, search_time

class NearPyBaseline(BaselineMethod):
    """NearPy LSH baseline"""
    
    def __init__(self, n_bits: int = 32, n_hashtables: int = 10):
        super().__init__(f'NearPy-{n_bits}b-{n_hashtables}t')
        self.n_bits = n_bits
        self.n_hashtables = n_hashtables
        
    def build_index(self, vectors: np.ndarray) -> float:
        try:
            from nearpy import Engine
            from nearpy.hashes import RandomBinaryProjections
        except ImportError:
            print("NearPy not available. Skipping NearPy baseline.")
            return 0.0
        
        start_time = time.time()
        
        d = vectors.shape[1]
        
        # Create LSH engine
        rbp = RandomBinaryProjections('rbp', self.n_bits)
        self.index = Engine(d, lshashes=[rbp])
        
        # Add vectors
        for i, vector in enumerate(vectors):
            self.index.store_vector(vector, i)
        
        return time.time() - start_time
    
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.index is None:
            return np.array([]), np.array([]), 0.0
        
        start_time = time.time()
        
        indices = []
        distances = []
        
        for query in queries:
            results = self.index.neighbours(query)
            
            if len(results) >= k:
                query_indices = [int(result[1]) for result in results[:k]]
                query_distances = [float(result[2]) for result in results[:k]]
            else:
                # Pad with random results if not enough found
                query_indices = [int(result[1]) for result in results]
                query_distances = [float(result[2]) for result in results]
                
                while len(query_indices) < k:
                    query_indices.append(np.random.randint(0, len(self.db_vectors)))
                    query_distances.append(0.0)
            
            indices.append(query_indices)
            distances.append(query_distances)
        
        search_time = time.time() - start_time
        
        return np.array(distances), np.array(indices), search_time

class ComparativeAnalysis:
    """Comprehensive comparison of Asteria against baselines"""
    
    def __init__(self, save_dir: str = "comparative_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results = {}
        
        # Setup color palette
        self.colors = plt.cm.Set2(np.linspace(0, 1, 10))
        sns.set_style("whitegrid")
        
    def setup_baselines(self) -> List[BaselineMethod]:
        """Setup all baseline methods"""
        baselines = []
        
        # FAISS variants
        baselines.append(FAISSBaseline('Flat'))
        baselines.append(FAISSBaseline('IVF'))
        baselines.append(FAISSBaseline('HNSW'))
        baselines.append(FAISSBaseline('PQ'))
        
        # Annoy variants
        baselines.append(AnnoyBaseline(50))
        baselines.append(AnnoyBaseline(100))
        
        # Other baselines
        baselines.append(BruteForceBaseline())
        baselines.append(NearPyBaseline(32, 10))
        
        return baselines
    
    def run_comprehensive_comparison(self, 
                                   db_vectors: np.ndarray,
                                   query_vectors: np.ndarray,
                                   dataset_name: str = "synthetic"):
        """Run comprehensive comparison on given dataset"""
        
        print(f"=== Running Comprehensive Comparison on {dataset_name} ===")
        print(f"Database size: {len(db_vectors)}")
        print(f"Query size: {len(query_vectors)}")
        print(f"Dimension: {db_vectors.shape[1]}")
        
        # Setup baselines
        baselines = self.setup_baselines()
        
        # Test different Asteria configurations
        asteria_configs = [
            {
                'name': 'Asteria-Fast',
                'raw_bits': 16, 'code_bits': 16, 'm_vantages': 32,
                'rank': 32, 'blocks': 8, 'target_mult': 4, 'max_radius': 1
            },
            {
                'name': 'Asteria-Balanced',
                'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48,
                'rank': 48, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
            },
            {
                'name': 'Asteria-Accurate',
                'raw_bits': 64, 'code_bits': 64, 'm_vantages': 96,
                'rank': 64, 'blocks': 16, 'target_mult': 12, 'max_radius': 3
            }
        ]
        
        results = []
        
        # Calculate ground truth for recall calculation
        print("Calculating ground truth...")
        gt_start = time.time()
        gt_similarities = np.dot(query_vectors, db_vectors.T)
        gt_indices = np.argpartition(-gt_similarities, 9, axis=1)[:, :10]
        gt_time = time.time() - gt_start
        print(f"Ground truth calculated in {gt_time:.2f}s")
        
        # Test Asteria configurations
        for config in asteria_configs:
            print(f"\nTesting {config['name']}...")
            
            try:
                # Create Asteria index
                dim = db_vectors.shape[1]
                bor = ButterflyRotation(dim)
                ecvh = ECVH(dim, config['m_vantages'], config['raw_bits'], config['code_bits'])
                lrsq = LRSQ(dim, config['rank'], config['blocks'])
                
                bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
                index = AsteriaIndexCPU(bundle, device='cpu')
                
                # Build index
                build_start = time.time()
                db_tensor = torch.tensor(db_vectors, dtype=torch.float32)
                index.add(db_tensor)
                build_time = time.time() - build_start
                
                # Search
                search_start = time.time()
                query_tensor = torch.tensor(query_vectors, dtype=torch.float32)
                distances, indices = index.search(
                    query_tensor, k=10,
                    target_mult=config['target_mult'],
                    max_radius=config['max_radius']
                )
                search_time = time.time() - search_start
                
                # Calculate metrics
                qps = len(query_vectors) / search_time
                recall = self._calculate_recall(indices.numpy(), gt_indices)
                
                # Estimate memory usage
                memory_mb = self._estimate_asteria_memory(config, len(db_vectors), dim)
                
                result = {
                    'method': config['name'],
                    'method_type': 'Asteria',
                    'build_time': build_time,
                    'search_time': search_time,
                    'qps': qps,
                    'recall@10': recall,
                    'memory_mb': memory_mb,
                    'config': config
                }
                
                results.append(result)
                
                print(f"  Build time: {build_time:.2f}s")
                print(f"  QPS: {qps:.2f}")
                print(f"  Recall@10: {recall:.4f}")
                print(f"  Memory: {memory_mb:.1f} MB")
                
            except Exception as e:
                print(f"  Error testing {config['name']}: {e}")
        
        # Test baseline methods
        for baseline in baselines:
            print(f"\nTesting {baseline.name}...")
            
            try:
                # Build index
                build_start = time.time()
                build_time = baseline.build_index(db_vectors)
                if build_time == 0.0:  # Method not available
                    continue
                    
                # Search
                distances, indices, search_time = baseline.search(query_vectors, k=10)
                
                if len(indices) == 0:  # Search failed
                    continue
                
                # Calculate metrics
                qps = len(query_vectors) / search_time
                recall = self._calculate_recall(indices, gt_indices)
                
                # Estimate memory usage
                memory_mb = self._estimate_baseline_memory(baseline.name, len(db_vectors), db_vectors.shape[1])
                
                result = {
                    'method': baseline.name,
                    'method_type': 'Baseline',
                    'build_time': build_time,
                    'search_time': search_time,
                    'qps': qps,
                    'recall@10': recall,
                    'memory_mb': memory_mb
                }
                
                results.append(result)
                
                print(f"  Build time: {build_time:.2f}s")
                print(f"  QPS: {qps:.2f}")
                print(f"  Recall@10: {recall:.4f}")
                print(f"  Memory: {memory_mb:.1f} MB")
                
            except Exception as e:
                print(f"  Error testing {baseline.name}: {e}")
        
        self.results[dataset_name] = results
        self._plot_comparison_results(results, dataset_name)
        return results
    
    def _calculate_recall(self, retrieved_indices: np.ndarray, gt_indices: np.ndarray) -> float:
        """Calculate recall@k"""
        recall_sum = 0
        for i in range(len(retrieved_indices)):
            retrieved_set = set(retrieved_indices[i])
            gt_set = set(gt_indices[i])
            recall_sum += len(retrieved_set & gt_set) / len(gt_set)
        
        return recall_sum / len(retrieved_indices)
    
    def _estimate_asteria_memory(self, config: Dict, n_vectors: int, dim: int) -> float:
        """Estimate Asteria memory usage in MB"""
        
        # ECVH hash tables
        ecvh_memory = config['m_vantages'] * config['raw_bits'] * n_vectors / 8  # bits to bytes
        
        # LRSQ compressed vectors
        lrsq_memory = n_vectors * config['rank'] * 4  # float32
        
        # BOR rotation matrix
        bor_memory = dim * dim * 4  # float32
        
        # Additional overhead
        overhead = 0.2 * (ecvh_memory + lrsq_memory + bor_memory)
        
        total_bytes = ecvh_memory + lrsq_memory + bor_memory + overhead
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _estimate_baseline_memory(self, method_name: str, n_vectors: int, dim: int) -> float:
        """Estimate baseline method memory usage in MB"""
        
        base_memory = n_vectors * dim * 4  # float32 vectors
        
        if 'FAISS-Flat' in method_name:
            return base_memory / (1024 * 1024)
        elif 'FAISS-IVF' in method_name:
            return base_memory * 1.1 / (1024 * 1024)  # 10% overhead
        elif 'FAISS-HNSW' in method_name:
            return base_memory * 1.5 / (1024 * 1024)  # 50% overhead for graph
        elif 'FAISS-PQ' in method_name:
            return base_memory * 0.25 / (1024 * 1024)  # ~75% compression
        elif 'Annoy' in method_name:
            return base_memory * 1.3 / (1024 * 1024)  # 30% overhead for trees
        elif 'BruteForce' in method_name:
            return base_memory / (1024 * 1024)
        elif 'NearPy' in method_name:
            return base_memory * 0.1 / (1024 * 1024)  # Only hash tables
        else:
            return base_memory / (1024 * 1024)
    
    def _plot_comparison_results(self, results: List[Dict], dataset_name: str):
        """Plot comprehensive comparison results"""
        
        if not results:
            print("No results to plot")
            return
        
        # Filter successful results
        valid_results = [r for r in results if r['qps'] > 0 and r['recall@10'] > 0]
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Comprehensive Comparison - {dataset_name.title()} Dataset', fontsize=16)
        
        methods = [r['method'] for r in valid_results]
        method_types = [r['method_type'] for r in valid_results]
        qps_values = [r['qps'] for r in valid_results]
        recall_values = [r['recall@10'] for r in valid_results]
        build_times = [r['build_time'] for r in valid_results]
        memory_values = [r['memory_mb'] for r in valid_results]
        
        # Color mapping
        colors = ['red' if t == 'Asteria' else 'blue' for t in method_types]
        
        # 1. QPS Comparison
        bars = axes[0, 0].bar(range(len(methods)), qps_values, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Queries Per Second (QPS)')
        axes[0, 0].set_title('Search Speed Comparison')
        axes[0, 0].set_xticks(range(len(methods)))
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, qps in zip(bars, qps_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(qps_values)*0.01,
                           f'{qps:.0f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Recall Comparison
        bars = axes[0, 1].bar(range(len(methods)), recall_values, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Recall@10')
        axes[0, 1].set_title('Accuracy Comparison')
        axes[0, 1].set_xticks(range(len(methods)))
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.1)
        
        # Add value labels
        for bar, recall in zip(bars, recall_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{recall:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Speed vs Accuracy
        scatter = axes[0, 2].scatter(recall_values, qps_values, c=colors, s=100, alpha=0.7)
        axes[0, 2].set_xlabel('Recall@10')
        axes[0, 2].set_ylabel('QPS')
        axes[0, 2].set_title('Speed vs Accuracy Tradeoff')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add method labels
        for i, method in enumerate(methods):
            axes[0, 2].annotate(method.split('-')[0], (recall_values[i], qps_values[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Build Time Comparison
        bars = axes[1, 0].bar(range(len(methods)), build_times, color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('Build Time (seconds)')
        axes[1, 0].set_title('Index Build Time')
        axes[1, 0].set_xticks(range(len(methods)))
        axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Memory Usage
        bars = axes[1, 1].bar(range(len(methods)), memory_values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Memory Usage (MB)')
        axes[1, 1].set_title('Memory Consumption')
        axes[1, 1].set_xticks(range(len(methods)))
        axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Efficiency Score (QPS / Memory)
        efficiency = [q / m if m > 0 else 0 for q, m in zip(qps_values, memory_values)]
        bars = axes[1, 2].bar(range(len(methods)), efficiency, color=colors, alpha=0.7)
        axes[1, 2].set_ylabel('Efficiency (QPS / MB)')
        axes[1, 2].set_title('Memory Efficiency')
        axes[1, 2].set_xticks(range(len(methods)))
        axes[1, 2].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Asteria'),
                          Patch(facecolor='blue', alpha=0.7, label='Baselines')]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{dataset_name}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_table(self, dataset_name: str) -> str:
        """Generate LaTeX performance table"""
        
        if dataset_name not in self.results:
            return ""
        
        results = self.results[dataset_name]
        valid_results = [r for r in results if r['qps'] > 0 and r['recall@10'] > 0]
        
        # Sort by QPS
        valid_results.sort(key=lambda x: x['qps'], reverse=True)
        
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Performance Comparison on " + dataset_name.title() + " Dataset}\n"
        latex += "\\begin{tabular}{|l|c|c|c|c|c|}\n"
        latex += "\\hline\n"
        latex += "Method & QPS & Recall@10 & Build Time (s) & Memory (MB) & Efficiency \\\\\n"
        latex += "\\hline\n"
        
        for result in valid_results:
            efficiency = result['qps'] / result['memory_mb'] if result['memory_mb'] > 0 else 0
            
            method_name = result['method'].replace('_', '\\_')
            latex += f"{method_name} & {result['qps']:.0f} & {result['recall@10']:.3f} & "
            latex += f"{result['build_time']:.1f} & {result['memory_mb']:.1f} & {efficiency:.2f} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        # Save to file
        with open(f'{self.save_dir}/{dataset_name}_table.tex', 'w') as f:
            f.write(latex)
        
        return latex
    
    def save_results(self, filename: str = 'comparative_results.json'):
        """Save all comparison results"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Comparison results saved to {filepath}")

def generate_test_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Generate test datasets of different characteristics"""
    
    datasets = {}
    
    # 1. Clustered data (image-like)
    print("Generating clustered dataset...")
    n_clusters = 50
    db_size = 20000
    query_size = 1000
    dim = 512
    
    cluster_centers = np.random.randn(n_clusters, dim)
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    
    db_vectors = []
    for i in range(db_size):
        cluster_id = np.random.randint(0, n_clusters)
        noise = np.random.randn(dim) * 0.2
        vector = cluster_centers[cluster_id] + noise
        vector = vector / np.linalg.norm(vector)
        db_vectors.append(vector)
    
    query_vectors = []
    for i in range(query_size):
        cluster_id = np.random.randint(0, n_clusters)
        noise = np.random.randn(dim) * 0.15
        vector = cluster_centers[cluster_id] + noise
        vector = vector / np.linalg.norm(vector)
        query_vectors.append(vector)
    
    datasets['clustered'] = (np.array(db_vectors), np.array(query_vectors))
    
    # 2. Uniform random data
    print("Generating uniform dataset...")
    db_uniform = np.random.randn(db_size, dim)
    db_uniform = db_uniform / np.linalg.norm(db_uniform, axis=1, keepdims=True)
    
    query_uniform = np.random.randn(query_size, dim)
    query_uniform = query_uniform / np.linalg.norm(query_uniform, axis=1, keepdims=True)
    
    datasets['uniform'] = (db_uniform, query_uniform)
    
    # 3. High-dimensional sparse data
    print("Generating sparse dataset...")
    sparse_dim = 1024
    sparsity = 0.1  # 10% non-zero elements
    
    db_sparse = np.random.randn(db_size, sparse_dim)
    mask = np.random.random((db_size, sparse_dim)) < sparsity
    db_sparse = db_sparse * mask
    db_sparse = db_sparse / (np.linalg.norm(db_sparse, axis=1, keepdims=True) + 1e-8)
    
    query_sparse = np.random.randn(query_size, sparse_dim)
    mask = np.random.random((query_size, sparse_dim)) < sparsity
    query_sparse = query_sparse * mask
    query_sparse = query_sparse / (np.linalg.norm(query_sparse, axis=1, keepdims=True) + 1e-8)
    
    datasets['sparse'] = (db_sparse, query_sparse)
    
    return datasets

def main():
    """Run comparative analysis"""
    
    print("Starting Comparative Analysis...")
    
    analysis = ComparativeAnalysis()
    
    # Generate test datasets
    datasets = generate_test_datasets()
    
    # Run comparisons on all datasets
    for dataset_name, (db_vectors, query_vectors) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Running comparison on {dataset_name} dataset")
        print(f"{'='*60}")
        
        analysis.run_comprehensive_comparison(db_vectors, query_vectors, dataset_name)
        
        # Generate LaTeX table
        latex_table = analysis.generate_performance_table(dataset_name)
        print(f"\nLaTeX table generated for {dataset_name}")
    
    # Save all results
    analysis.save_results()
    
    print(f"\nComparative analysis completed! Results saved in '{analysis.save_dir}' directory.")
    print("Generated files:")
    print("  - comparative_results.json (all results)")
    print("  - *_comparison.png (comparison plots)")
    print("  - *_table.tex (LaTeX tables)")

if __name__ == "__main__":
    main()
