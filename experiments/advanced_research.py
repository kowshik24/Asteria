"""
Advanced research experiments for Asteria
Including dimensionality analysis, parameter studies, and novel metrics
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

class AdvancedResearchExperiments:
    """Advanced experiments for research paper"""
    
    def __init__(self, save_dir: str = "advanced_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results = {}
        
        # Setup plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def run_dimensionality_study(self, 
                                dims: List[int] = [128, 256, 512, 768, 1024, 1536, 2048],
                                db_size: int = 10000):
        """Study performance across different dimensions"""
        
        print("=== Running Dimensionality Study ===")
        
        results = []
        
        for dim in dims:
            print(f"\nTesting dimension: {dim}")
            
            # Generate synthetic data
            db_vectors = np.random.randn(db_size, dim)
            db_vectors = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
            
            query_vectors = np.random.randn(1000, dim)
            query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
            
            # Ground truth
            gt_similarities = np.dot(query_vectors, db_vectors.T)
            gt_indices = np.argpartition(-gt_similarities, 9, axis=1)[:, :10]
            
            # Test different configurations
            configs = [
                {
                    'name': 'Fast',
                    'raw_bits': max(8, min(32, dim // 16)),
                    'code_bits': max(8, min(32, dim // 16)),
                    'm_vantages': max(16, min(64, dim // 8)),
                    'rank': max(16, min(64, dim // 8)),
                    'blocks': max(4, min(16, dim // 32)),
                    'target_mult': 4, 'max_radius': 1
                },
                {
                    'name': 'Balanced',
                    'raw_bits': max(16, min(64, dim // 8)),
                    'code_bits': max(16, min(64, dim // 8)),
                    'm_vantages': max(32, min(96, dim // 4)),
                    'rank': max(32, min(96, dim // 4)),
                    'blocks': max(8, min(24, dim // 16)),
                    'target_mult': 8, 'max_radius': 2
                }
            ]
            
            for config in configs:
                try:
                    # Create index
                    bor = ButterflyRotation(dim)
                    ecvh = ECVH(dim, config['m_vantages'], config['raw_bits'], config['code_bits'])
                    lrsq = LRSQ(dim, config['rank'], config['blocks'])
                    
                    bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
                    index = AsteriaIndexCPU(bundle, device='cpu')
                    
                    # Build and search
                    build_start = time.time()
                    db_tensor = torch.tensor(db_vectors, dtype=torch.float32)
                    index.add(db_tensor)
                    build_time = time.time() - build_start
                    
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
                    
                    # Memory estimation
                    memory_mb = self._estimate_memory(config, db_size, dim)
                    
                    result = {
                        'dimension': dim,
                        'config': config['name'],
                        'qps': qps,
                        'recall@10': recall,
                        'build_time': build_time,
                        'memory_mb': memory_mb,
                        'db_size': db_size,
                        'compression_ratio': (db_size * dim * 4) / (memory_mb * 1024 * 1024)
                    }
                    
                    results.append(result)
                    
                    print(f"  {config['name']}: QPS={qps:.1f}, Recall={recall:.3f}")
                    
                except Exception as e:
                    print(f"  Error with {config['name']} at dim {dim}: {e}")
        
        self.results['dimensionality_study'] = results
        self._plot_dimensionality_results(results)
        return results
    
    def run_parameter_sensitivity_analysis(self, dim: int = 512, db_size: int = 10000):
        """Analyze sensitivity to different parameters"""
        
        print("=== Running Parameter Sensitivity Analysis ===")
        
        # Generate test data
        db_vectors = np.random.randn(db_size, dim)
        db_vectors = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
        
        query_vectors = np.random.randn(1000, dim)
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        
        # Ground truth
        gt_similarities = np.dot(query_vectors, db_vectors.T)
        gt_indices = np.argpartition(-gt_similarities, 9, axis=1)[:, :10]
        
        all_results = {}
        
        # 1. Hash bits sensitivity
        print("\n1. Testing hash bits sensitivity...")
        hash_bits_results = []
        for bits in [8, 16, 24, 32, 48, 64]:
            print(f"  Testing {bits} bits...")
            
            config = {
                'raw_bits': bits, 'code_bits': bits, 'm_vantages': 48,
                'rank': 48, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
            }
            
            result = self._test_single_config(config, db_vectors, query_vectors, gt_indices)
            if result:
                result['parameter'] = 'hash_bits'
                result['value'] = bits
                hash_bits_results.append(result)
        
        all_results['hash_bits'] = hash_bits_results
        
        # 2. Vantage points sensitivity
        print("\n2. Testing vantage points sensitivity...")
        vantage_results = []
        for vantages in [16, 32, 48, 64, 96, 128]:
            print(f"  Testing {vantages} vantages...")
            
            config = {
                'raw_bits': 32, 'code_bits': 32, 'm_vantages': vantages,
                'rank': 48, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
            }
            
            result = self._test_single_config(config, db_vectors, query_vectors, gt_indices)
            if result:
                result['parameter'] = 'vantages'
                result['value'] = vantages
                vantage_results.append(result)
        
        all_results['vantages'] = vantage_results
        
        # 3. LRSQ rank sensitivity
        print("\n3. Testing LRSQ rank sensitivity...")
        rank_results = []
        for rank in [16, 32, 48, 64, 96, 128]:
            print(f"  Testing rank {rank}...")
            
            config = {
                'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48,
                'rank': rank, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
            }
            
            result = self._test_single_config(config, db_vectors, query_vectors, gt_indices)
            if result:
                result['parameter'] = 'rank'
                result['value'] = rank
                rank_results.append(result)
        
        all_results['rank'] = rank_results
        
        # 4. Search parameters sensitivity
        print("\n4. Testing search parameters sensitivity...")
        search_results = []
        for target_mult in [2, 4, 8, 12, 16, 24]:
            for max_radius in [1, 2, 3, 4]:
                print(f"  Testing target_mult={target_mult}, max_radius={max_radius}...")
                
                config = {
                    'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48,
                    'rank': 48, 'blocks': 12, 'target_mult': target_mult, 'max_radius': max_radius
                }
                
                result = self._test_single_config(config, db_vectors, query_vectors, gt_indices)
                if result:
                    result['parameter'] = 'search_params'
                    result['value'] = f"{target_mult}x{max_radius}"
                    result['target_mult'] = target_mult
                    result['max_radius'] = max_radius
                    search_results.append(result)
        
        all_results['search_params'] = search_results
        
        self.results['parameter_sensitivity'] = all_results
        self._plot_parameter_sensitivity(all_results)
        return all_results
    
    def run_scalability_analysis(self, max_size: int = 100000):
        """Comprehensive scalability analysis"""
        
        print("=== Running Scalability Analysis ===")
        
        dim = 512
        db_sizes = [1000, 2500, 5000, 10000, 25000, 50000]
        if max_size > 50000:
            db_sizes.append(max_size)
        
        results = []
        
        for db_size in db_sizes:
            print(f"\nTesting database size: {db_size}")
            
            # Generate data
            db_vectors = np.random.randn(db_size, dim)
            db_vectors = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
            
            query_size = min(1000, db_size // 10)
            query_vectors = np.random.randn(query_size, dim)
            query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
            
            # Ground truth (sample for large datasets)
            if db_size > 20000:
                # Sample for ground truth calculation
                sample_indices = np.random.choice(db_size, 10000, replace=False)
                gt_similarities = np.dot(query_vectors, db_vectors[sample_indices].T)
                gt_sample_indices = np.argpartition(-gt_similarities, 9, axis=1)[:, :10]
                gt_indices = sample_indices[gt_sample_indices]
            else:
                gt_similarities = np.dot(query_vectors, db_vectors.T)
                gt_indices = np.argpartition(-gt_similarities, 9, axis=1)[:, :10]
            
            # Test balanced configuration
            config = {
                'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48,
                'rank': 48, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
            }
            
            try:
                # Create index
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
                
                # Calculate recall (approximate for large datasets)
                if db_size > 20000:
                    # Approximate recall using sample
                    recall = self._approximate_recall(indices.numpy(), gt_indices, sample_indices)
                else:
                    recall = self._calculate_recall(indices.numpy(), gt_indices)
                
                memory_mb = self._estimate_memory(config, db_size, dim)
                throughput_mb_s = (query_size * dim * 4) / (search_time * 1024 * 1024)
                
                result = {
                    'db_size': db_size,
                    'query_size': query_size,
                    'qps': qps,
                    'recall@10': recall,
                    'build_time': build_time,
                    'search_time': search_time,
                    'memory_mb': memory_mb,
                    'throughput_mb_s': throughput_mb_s,
                    'build_rate_vectors_s': db_size / build_time,
                    'memory_per_vector_bytes': (memory_mb * 1024 * 1024) / db_size
                }
                
                results.append(result)
                
                print(f"  QPS: {qps:.1f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  Build time: {build_time:.1f}s")
                print(f"  Memory: {memory_mb:.1f} MB")
                
            except Exception as e:
                print(f"  Error at size {db_size}: {e}")
        
        self.results['scalability_analysis'] = results
        self._plot_scalability_analysis(results)
        return results
    
    def run_clustering_analysis(self, n_clusters_list: List[int] = [10, 25, 50, 100, 200]):
        """Analyze performance on clustered data with different cluster counts"""
        
        print("=== Running Clustering Analysis ===")
        
        dim = 512
        db_size = 20000
        query_size = 1000
        
        results = []
        
        for n_clusters in n_clusters_list:
            print(f"\nTesting {n_clusters} clusters...")
            
            # Generate clustered data
            cluster_centers = np.random.randn(n_clusters, dim)
            cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
            
            # Generate database
            db_vectors = []
            db_labels = []
            for i in range(db_size):
                cluster_id = np.random.randint(0, n_clusters)
                noise_std = 0.2 if n_clusters <= 50 else 0.1  # Less noise for more clusters
                noise = np.random.randn(dim) * noise_std
                vector = cluster_centers[cluster_id] + noise
                vector = vector / np.linalg.norm(vector)
                db_vectors.append(vector)
                db_labels.append(cluster_id)
            
            db_vectors = np.array(db_vectors)
            db_labels = np.array(db_labels)
            
            # Generate queries
            query_vectors = []
            query_labels = []
            for i in range(query_size):
                cluster_id = np.random.randint(0, n_clusters)
                noise = np.random.randn(dim) * 0.15
                vector = cluster_centers[cluster_id] + noise
                vector = vector / np.linalg.norm(vector)
                query_vectors.append(vector)
                query_labels.append(cluster_id)
            
            query_vectors = np.array(query_vectors)
            query_labels = np.array(query_labels)
            
            # Ground truth
            gt_similarities = np.dot(query_vectors, db_vectors.T)
            gt_indices = np.argpartition(-gt_similarities, 9, axis=1)[:, :10]
            
            # Test configuration
            config = {
                'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48,
                'rank': 48, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
            }
            
            try:
                # Create and test index
                bor = ButterflyRotation(dim)
                ecvh = ECVH(dim, config['m_vantages'], config['raw_bits'], config['code_bits'])
                lrsq = LRSQ(dim, config['rank'], config['blocks'])
                
                bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
                index = AsteriaIndexCPU(bundle, device='cpu')
                
                # Build and search
                build_start = time.time()
                db_tensor = torch.tensor(db_vectors, dtype=torch.float32)
                index.add(db_tensor)
                build_time = time.time() - build_start
                
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
                semantic_recall = self._calculate_semantic_recall(
                    indices.numpy(), query_labels, db_labels)
                
                # Cluster quality metrics
                cluster_purity = self._calculate_cluster_purity(indices.numpy(), query_labels, db_labels)
                
                result = {
                    'n_clusters': n_clusters,
                    'qps': qps,
                    'recall@10': recall,
                    'semantic_recall@10': semantic_recall,
                    'cluster_purity': cluster_purity,
                    'build_time': build_time,
                    'avg_cluster_size': db_size / n_clusters,
                    'cluster_separation': self._calculate_cluster_separation(cluster_centers)
                }
                
                results.append(result)
                
                print(f"  QPS: {qps:.1f}")
                print(f"  Standard Recall: {recall:.3f}")
                print(f"  Semantic Recall: {semantic_recall:.3f}")
                print(f"  Cluster Purity: {cluster_purity:.3f}")
                
            except Exception as e:
                print(f"  Error with {n_clusters} clusters: {e}")
        
        self.results['clustering_analysis'] = results
        self._plot_clustering_analysis(results)
        return results
    
    def _test_single_config(self, config: Dict, db_vectors: np.ndarray, 
                           query_vectors: np.ndarray, gt_indices: np.ndarray) -> Optional[Dict]:
        """Test a single configuration"""
        
        try:
            dim = db_vectors.shape[1]
            
            # Create index
            bor = ButterflyRotation(dim)
            ecvh = ECVH(dim, config['m_vantages'], config['raw_bits'], config['code_bits'])
            lrsq = LRSQ(dim, config['rank'], config['blocks'])
            
            bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
            index = AsteriaIndexCPU(bundle, device='cpu')
            
            # Build
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
            
            # Metrics
            qps = len(query_vectors) / search_time
            recall = self._calculate_recall(indices.numpy(), gt_indices)
            memory_mb = self._estimate_memory(config, len(db_vectors), dim)
            
            return {
                'qps': qps,
                'recall@10': recall,
                'build_time': build_time,
                'memory_mb': memory_mb,
                'config': config
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def _calculate_recall(self, retrieved_indices: np.ndarray, gt_indices: np.ndarray) -> float:
        """Calculate recall@k"""
        recall_sum = 0
        for i in range(len(retrieved_indices)):
            retrieved_set = set(retrieved_indices[i])
            gt_set = set(gt_indices[i])
            recall_sum += len(retrieved_set & gt_set) / len(gt_set)
        
        return recall_sum / len(retrieved_indices)
    
    def _approximate_recall(self, retrieved_indices: np.ndarray, 
                           gt_indices: np.ndarray, sample_indices: np.ndarray) -> float:
        """Approximate recall for large datasets using sampling"""
        recall_sum = 0
        for i in range(len(retrieved_indices)):
            # Count how many retrieved indices are in the sample ground truth
            retrieved_in_sample = np.intersect1d(retrieved_indices[i], sample_indices)
            gt_in_sample = gt_indices[i]
            
            if len(gt_in_sample) > 0:
                overlap = len(np.intersect1d(retrieved_in_sample, gt_in_sample))
                recall_sum += overlap / len(gt_in_sample)
        
        return recall_sum / len(retrieved_indices)
    
    def _calculate_semantic_recall(self, retrieved_indices: np.ndarray, 
                                 query_labels: np.ndarray, db_labels: np.ndarray) -> float:
        """Calculate semantic recall (same class)"""
        recall_sum = 0
        for i in range(len(retrieved_indices)):
            query_class = query_labels[i]
            retrieved_classes = db_labels[retrieved_indices[i]]
            same_class_count = np.sum(retrieved_classes == query_class)
            recall_sum += same_class_count / 10  # k=10
        
        return recall_sum / len(retrieved_indices)
    
    def _calculate_cluster_purity(self, retrieved_indices: np.ndarray,
                                query_labels: np.ndarray, db_labels: np.ndarray) -> float:
        """Calculate cluster purity of retrieved results"""
        purity_sum = 0
        for i in range(len(retrieved_indices)):
            retrieved_classes = db_labels[retrieved_indices[i]]
            # Find most common class
            unique_classes, counts = np.unique(retrieved_classes, return_counts=True)
            max_count = np.max(counts)
            purity_sum += max_count / len(retrieved_classes)
        
        return purity_sum / len(retrieved_indices)
    
    def _calculate_cluster_separation(self, cluster_centers: np.ndarray) -> float:
        """Calculate average separation between cluster centers"""
        distances = []
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                distances.append(dist)
        
        return np.mean(distances)
    
    def _estimate_memory(self, config: Dict, n_vectors: int, dim: int) -> float:
        """Estimate memory usage in MB"""
        
        # ECVH hash tables
        ecvh_memory = config['m_vantages'] * config['raw_bits'] * n_vectors / 8
        
        # LRSQ compressed vectors
        lrsq_memory = n_vectors * config['rank'] * 4
        
        # BOR rotation matrix
        bor_memory = dim * dim * 4
        
        # Overhead
        overhead = 0.2 * (ecvh_memory + lrsq_memory + bor_memory)
        
        total_bytes = ecvh_memory + lrsq_memory + bor_memory + overhead
        return total_bytes / (1024 * 1024)
    
    def _plot_dimensionality_results(self, results: List[Dict]):
        """Plot dimensionality study results"""
        
        # Group by config
        configs = {}
        for result in results:
            config_name = result['config']
            if config_name not in configs:
                configs[config_name] = {'dims': [], 'qps': [], 'recall': [], 'memory': [], 'compression': []}
            
            configs[config_name]['dims'].append(result['dimension'])
            configs[config_name]['qps'].append(result['qps'])
            configs[config_name]['recall'].append(result['recall@10'])
            configs[config_name]['memory'].append(result['memory_mb'])
            configs[config_name]['compression'].append(result['compression_ratio'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dimensionality Study Results', fontsize=16)
        
        # QPS vs Dimension
        for config_name, data in configs.items():
            axes[0, 0].plot(data['dims'], data['qps'], 'o-', label=config_name, linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('QPS')
        axes[0, 0].set_title('Search Speed vs Dimension')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall vs Dimension
        for config_name, data in configs.items():
            axes[0, 1].plot(data['dims'], data['recall'], 'o-', label=config_name, linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Recall@10')
        axes[0, 1].set_title('Accuracy vs Dimension')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.1)
        
        # Memory vs Dimension
        for config_name, data in configs.items():
            axes[1, 0].plot(data['dims'], data['memory'], 'o-', label=config_name, linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].set_title('Memory Usage vs Dimension')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Compression Ratio vs Dimension
        for config_name, data in configs.items():
            axes[1, 1].plot(data['dims'], data['compression'], 'o-', label=config_name, linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Compression Ratio')
        axes[1, 1].set_title('Compression vs Dimension')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/dimensionality_study.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_parameter_sensitivity(self, all_results: Dict):
        """Plot parameter sensitivity analysis"""
        
        n_params = len(all_results)
        fig, axes = plt.subplots(2, n_params, figsize=(5*n_params, 10))
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
        
        if n_params == 1:
            axes = axes.reshape(2, 1)
        
        for i, (param_name, results) in enumerate(all_results.items()):
            if not results:
                continue
            
            if param_name == 'search_params':
                # Special handling for search parameters
                target_mults = sorted(set(r['target_mult'] for r in results))
                max_radii = sorted(set(r['max_radius'] for r in results))
                
                # Create heatmap data
                qps_matrix = np.zeros((len(max_radii), len(target_mults)))
                recall_matrix = np.zeros((len(max_radii), len(target_mults)))
                
                for result in results:
                    row_idx = max_radii.index(result['max_radius'])
                    col_idx = target_mults.index(result['target_mult'])
                    qps_matrix[row_idx, col_idx] = result['qps']
                    recall_matrix[row_idx, col_idx] = result['recall@10']
                
                # QPS heatmap
                im1 = axes[0, i].imshow(qps_matrix, cmap='viridis', aspect='auto')
                axes[0, i].set_xticks(range(len(target_mults)))
                axes[0, i].set_xticklabels(target_mults)
                axes[0, i].set_yticks(range(len(max_radii)))
                axes[0, i].set_yticklabels(max_radii)
                axes[0, i].set_xlabel('Target Multiplier')
                axes[0, i].set_ylabel('Max Radius')
                axes[0, i].set_title(f'QPS - {param_name}')
                plt.colorbar(im1, ax=axes[0, i])
                
                # Recall heatmap
                im2 = axes[1, i].imshow(recall_matrix, cmap='plasma', aspect='auto')
                axes[1, i].set_xticks(range(len(target_mults)))
                axes[1, i].set_xticklabels(target_mults)
                axes[1, i].set_yticks(range(len(max_radii)))
                axes[1, i].set_yticklabels(max_radii)
                axes[1, i].set_xlabel('Target Multiplier')
                axes[1, i].set_ylabel('Max Radius')
                axes[1, i].set_title(f'Recall@10 - {param_name}')
                plt.colorbar(im2, ax=axes[1, i])
                
            else:
                # Regular line plots
                values = [r['value'] for r in results]
                qps_vals = [r['qps'] for r in results]
                recall_vals = [r['recall@10'] for r in results]
                
                # Sort by parameter value
                sorted_data = sorted(zip(values, qps_vals, recall_vals))
                values, qps_vals, recall_vals = zip(*sorted_data)
                
                # QPS plot
                axes[0, i].plot(values, qps_vals, 'o-', linewidth=2, markersize=6)
                axes[0, i].set_xlabel(param_name.replace('_', ' ').title())
                axes[0, i].set_ylabel('QPS')
                axes[0, i].set_title(f'QPS vs {param_name}')
                axes[0, i].grid(True, alpha=0.3)
                
                # Recall plot
                axes[1, i].plot(values, recall_vals, 'o-', linewidth=2, markersize=6, color='red')
                axes[1, i].set_xlabel(param_name.replace('_', ' ').title())
                axes[1, i].set_ylabel('Recall@10')
                axes[1, i].set_title(f'Recall vs {param_name}')
                axes[1, i].grid(True, alpha=0.3)
                axes[1, i].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_scalability_analysis(self, results: List[Dict]):
        """Plot scalability analysis results"""
        
        db_sizes = [r['db_size'] for r in results]
        qps_values = [r['qps'] for r in results]
        recall_values = [r['recall@10'] for r in results]
        build_times = [r['build_time'] for r in results]
        memory_values = [r['memory_mb'] for r in results]
        build_rates = [r['build_rate_vectors_s'] for r in results]
        memory_per_vector = [r['memory_per_vector_bytes'] for r in results]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Scalability Analysis', fontsize=16)
        
        # QPS vs Database Size
        axes[0, 0].loglog(db_sizes, qps_values, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Database Size')
        axes[0, 0].set_ylabel('QPS')
        axes[0, 0].set_title('Search Speed Scalability')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Build Time vs Database Size
        axes[0, 1].loglog(db_sizes, build_times, 'o-', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Database Size')
        axes[0, 1].set_ylabel('Build Time (seconds)')
        axes[0, 1].set_title('Index Construction Scalability')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Memory vs Database Size
        axes[0, 2].loglog(db_sizes, memory_values, 'o-', linewidth=2, markersize=8, color='purple')
        axes[0, 2].set_xlabel('Database Size')
        axes[0, 2].set_ylabel('Memory Usage (MB)')
        axes[0, 2].set_title('Memory Scalability')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Recall vs Database Size
        axes[1, 0].semilogx(db_sizes, recall_values, 'o-', linewidth=2, markersize=8, color='red')
        axes[1, 0].set_xlabel('Database Size')
        axes[1, 0].set_ylabel('Recall@10')
        axes[1, 0].set_title('Accuracy vs Scale')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1.1)
        
        # Build Rate (vectors/second)
        axes[1, 1].semilogx(db_sizes, build_rates, 'o-', linewidth=2, markersize=8, color='orange')
        axes[1, 1].set_xlabel('Database Size')
        axes[1, 1].set_ylabel('Build Rate (vectors/second)')
        axes[1, 1].set_title('Index Construction Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Memory per Vector
        axes[1, 2].semilogx(db_sizes, memory_per_vector, 'o-', linewidth=2, markersize=8, color='brown')
        axes[1, 2].set_xlabel('Database Size')
        axes[1, 2].set_ylabel('Memory per Vector (bytes)')
        axes[1, 2].set_title('Memory Efficiency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_clustering_analysis(self, results: List[Dict]):
        """Plot clustering analysis results"""
        
        n_clusters = [r['n_clusters'] for r in results]
        qps_values = [r['qps'] for r in results]
        recall_values = [r['recall@10'] for r in results]
        semantic_recall = [r['semantic_recall@10'] for r in results]
        cluster_purity = [r['cluster_purity'] for r in results]
        cluster_sizes = [r['avg_cluster_size'] for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clustering Analysis Results', fontsize=16)
        
        # QPS vs Number of Clusters
        axes[0, 0].plot(n_clusters, qps_values, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('QPS')
        axes[0, 0].set_title('Search Speed vs Cluster Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall Comparison
        axes[0, 1].plot(n_clusters, recall_values, 'o-', label='Standard Recall@10', linewidth=2, markersize=8)
        axes[0, 1].plot(n_clusters, semantic_recall, 's-', label='Semantic Recall@10', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Recall@10')
        axes[0, 1].set_title('Recall vs Cluster Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.1)
        
        # Cluster Purity
        axes[1, 0].plot(n_clusters, cluster_purity, 'o-', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Cluster Purity')
        axes[1, 0].set_title('Result Quality vs Cluster Count')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1.1)
        
        # Average Cluster Size Effect
        axes[1, 1].plot(cluster_sizes, semantic_recall, 'o-', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_xlabel('Average Cluster Size')
        axes[1, 1].set_ylabel('Semantic Recall@10')
        axes[1, 1].set_title('Semantic Recall vs Cluster Size')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str = 'advanced_results.json'):
        """Save all experimental results"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Advanced experimental results saved to {filepath}")
    
    def generate_research_summary(self):
        """Generate comprehensive research summary"""
        
        print("\n" + "="*80)
        print("ADVANCED RESEARCH EXPERIMENTS SUMMARY")
        print("="*80)
        
        for experiment_name in self.results:
            print(f"\n{experiment_name.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            if experiment_name == 'dimensionality_study':
                results = self.results[experiment_name]
                configs = {}
                for r in results:
                    if r['config'] not in configs:
                        configs[r['config']] = []
                    configs[r['config']].append(r)
                
                for config_name, config_results in configs.items():
                    print(f"\n  {config_name} Configuration:")
                    print(f"    Dimension range: {min(r['dimension'] for r in config_results)}-{max(r['dimension'] for r in config_results)}")
                    print(f"    QPS range: {min(r['qps'] for r in config_results):.1f}-{max(r['qps'] for r in config_results):.1f}")
                    print(f"    Recall range: {min(r['recall@10'] for r in config_results):.3f}-{max(r['recall@10'] for r in config_results):.3f}")
                    print(f"    Max compression: {max(r['compression_ratio'] for r in config_results):.1f}x")
            
            elif experiment_name == 'scalability_analysis':
                results = self.results[experiment_name]
                print(f"    Database size range: {min(r['db_size'] for r in results)}-{max(r['db_size'] for r in results):,}")
                print(f"    QPS range: {min(r['qps'] for r in results):.1f}-{max(r['qps'] for r in results):.1f}")
                print(f"    Recall range: {min(r['recall@10'] for r in results):.3f}-{max(r['recall@10'] for r in results):.3f}")
                print(f"    Memory efficiency: {min(r['memory_per_vector_bytes'] for r in results):.1f}-{max(r['memory_per_vector_bytes'] for r in results):.1f} bytes/vector")
            
            elif experiment_name == 'clustering_analysis':
                results = self.results[experiment_name]
                print(f"    Cluster range: {min(r['n_clusters'] for r in results)}-{max(r['n_clusters'] for r in results)}")
                print(f"    Best semantic recall: {max(r['semantic_recall@10'] for r in results):.3f}")
                print(f"    Best cluster purity: {max(r['cluster_purity'] for r in results):.3f}")
            
            elif experiment_name == 'parameter_sensitivity':
                for param_name, param_results in self.results[experiment_name].items():
                    if param_results:
                        print(f"\n  {param_name.replace('_', ' ').title()}:")
                        print(f"    QPS range: {min(r['qps'] for r in param_results):.1f}-{max(r['qps'] for r in param_results):.1f}")
                        print(f"    Recall range: {min(r['recall@10'] for r in param_results):.3f}-{max(r['recall@10'] for r in param_results):.3f}")

def main():
    """Run advanced research experiments"""
    
    print("Starting Advanced Research Experiments for Asteria...")
    
    experiments = AdvancedResearchExperiments()
    
    # 1. Dimensionality study
    print("\n1. Running dimensionality study...")
    experiments.run_dimensionality_study()
    
    # 2. Parameter sensitivity analysis
    print("\n2. Running parameter sensitivity analysis...")
    experiments.run_parameter_sensitivity_analysis()
    
    # 3. Scalability analysis
    print("\n3. Running scalability analysis...")
    experiments.run_scalability_analysis(max_size=75000)
    
    # 4. Clustering analysis
    print("\n4. Running clustering analysis...")
    experiments.run_clustering_analysis()
    
    # Save results and generate summary
    experiments.save_results()
    experiments.generate_research_summary()
    
    print(f"\nAdvanced research experiments completed!")
    print(f"Results saved in '{experiments.save_dir}' directory.")
    print("\nGenerated files:")
    print("  - advanced_results.json (all experimental data)")
    print("  - dimensionality_study.png (dimension analysis)")
    print("  - parameter_sensitivity.png (parameter studies)")
    print("  - scalability_analysis.png (scalability results)")
    print("  - clustering_analysis.png (clustering performance)")

if __name__ == "__main__":
    main()
