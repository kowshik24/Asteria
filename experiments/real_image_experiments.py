"""
Real image dataset experiments for Asteria research
Support for CIFAR-10, ImageNet, and custom image datasets
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Asteria imports
from asteria.bor import ButterflyRotation
from asteria.ecvh import ECVH
from asteria.lrsq import LRSQ
from asteria.index_cpu import AsteriaIndexCPU

# Import our custom modules
from experiments.image_features import ImageFeatureExtractor, create_synthetic_image_dataset

class RealImageExperiment:
    """Experiments on real image datasets"""
    
    def __init__(self, save_dir: str = "real_image_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results = {}
        
    def setup_cifar10_experiment(self, 
                                subset_size: Optional[int] = None,
                                feature_model: str = 'resnet') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Setup CIFAR-10 dataset experiment"""
        
        print("Setting up CIFAR-10 experiment...")
        
        # Use smaller default subset for faster experimentation
        if subset_size is None:
            subset_size = 5000  # Much smaller default for faster testing
        
        try:
            import torchvision
            import torchvision.transforms as transforms
            
            print(f"Will process {subset_size} training images and {min(500, subset_size//10)} test images")
            
            # Download CIFAR-10
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                                  download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                                 download=True, transform=transform)
            
            # Extract features with auto device detection
            extractor = ImageFeatureExtractor(feature_model, device='auto')
            
            # Process training set (database) with batch processing
            train_features = []
            train_labels = []
            
            subset_size_train = subset_size if subset_size else len(trainset)
            batch_size = 64  # Process multiple images at once
            
            print(f"Extracting features from {subset_size_train} training images with batch size {batch_size}...")
            
            for i in range(0, min(subset_size_train, len(trainset)), batch_size):
                batch_end = min(i + batch_size, min(subset_size_train, len(trainset)))
                batch_images = []
                batch_labels = []
                
                # Collect batch
                for j in range(i, batch_end):
                    img, label = trainset[j]
                    batch_images.append(img)
                    batch_labels.append(label)
                
                # Process batch
                if batch_images:
                    batch_tensor = torch.stack(batch_images)
                    batch_features = extractor._extract_batch(batch_tensor)
                    
                    # Store results
                    for k, features in enumerate(batch_features):
                        train_features.append(features.cpu().numpy())
                        train_labels.append(batch_labels[k])
                
                if i % 1000 == 0:
                    print(f"Processed {i}/{subset_size_train} training images")
            
            # Process test set (queries) with batch processing
            test_features = []
            test_labels = []
            
            subset_size_test = min(500, len(testset)) if subset_size else len(testset)
            
            print(f"Extracting features from {subset_size_test} test images with batch size {batch_size}...")
            
            for i in range(0, subset_size_test, batch_size):
                batch_end = min(i + batch_size, subset_size_test)
                batch_images = []
                batch_labels = []
                
                # Collect batch
                for j in range(i, batch_end):
                    img, label = testset[j]
                    batch_images.append(img)
                    batch_labels.append(label)
                
                # Process batch
                if batch_images:
                    batch_tensor = torch.stack(batch_images)
                    batch_features = extractor._extract_batch(batch_tensor)
                    
                    # Store results
                    for k, features in enumerate(batch_features):
                        test_features.append(features.cpu().numpy())
                        test_labels.append(batch_labels[k])
                
                if i % 100 == 0:
                    print(f"Processed {i}/{subset_size_test} test images")
            
            db_features = np.vstack(train_features)
            query_features = np.vstack(test_features)
            
            # Normalize features
            db_features = db_features / np.linalg.norm(db_features, axis=1, keepdims=True)
            query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
            
            print(f"CIFAR-10 setup complete:")
            print(f"  Database: {db_features.shape}")
            print(f"  Queries: {query_features.shape}")
            
            return db_features, query_features, np.array(test_labels)
            
        except ImportError:
            print("torchvision not available. Generating synthetic image-like data...")
            return self._generate_synthetic_cifar10_like(subset_size)
    
    def _generate_synthetic_cifar10_like(self, subset_size: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data that mimics CIFAR-10 structure"""
        
        db_size = subset_size if subset_size else 50000
        query_size = min(1000, db_size // 10)
        
        print(f"Generating synthetic CIFAR-10-like data...")
        print(f"  Database size: {db_size}")
        print(f"  Query size: {query_size}")
        
        # Create clustered data (10 clusters for 10 classes)
        n_clusters = 10
        dim = 512  # ResNet-like feature dimension
        
        # Generate cluster centers
        cluster_centers = np.random.randn(n_clusters, dim)
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        
        # Generate database
        db_features = []
        db_labels = []
        
        for i in range(db_size):
            cluster_id = np.random.randint(0, n_clusters)
            # Add noise to cluster center
            noise = np.random.randn(dim) * 0.3
            feature = cluster_centers[cluster_id] + noise
            feature = feature / np.linalg.norm(feature)
            
            db_features.append(feature)
            db_labels.append(cluster_id)
        
        # Generate queries
        query_features = []
        query_labels = []
        
        for i in range(query_size):
            cluster_id = np.random.randint(0, n_clusters)
            noise = np.random.randn(dim) * 0.2  # Less noise for queries
            feature = cluster_centers[cluster_id] + noise
            feature = feature / np.linalg.norm(feature)
            
            query_features.append(feature)
            query_labels.append(cluster_id)
        
        return np.array(db_features), np.array(query_features), np.array(query_labels)
    
    def run_cifar10_experiment(self, 
                              subset_size: Optional[int] = 10000,
                              feature_model: str = 'resnet'):
        """Run complete CIFAR-10 experiment"""
        
        print("=== Running CIFAR-10 Experiment ===")
        
        # Setup dataset
        db_features, query_features, query_labels = self.setup_cifar10_experiment(
            subset_size, feature_model)
        
        # Different Asteria configurations to test
        configs = [
            {
                'name': 'Fast',
                'raw_bits': 16, 'code_bits': 16, 'm_vantages': 32,
                'rank': 32, 'blocks': 8, 'target_mult': 4, 'max_radius': 1
            },
            {
                'name': 'Balanced',
                'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48,
                'rank': 48, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
            },
            {
                'name': 'Accurate',
                'raw_bits': 48, 'code_bits': 64, 'm_vantages': 96,
                'rank': 64, 'blocks': 16, 'target_mult': 12, 'max_radius': 3
            }
        ]
        
        results = []
        
        for config in configs:
            print(f"\nTesting configuration: {config['name']}")
            
            # Create Asteria index
            dim = db_features.shape[1]
            bor = ButterflyRotation(dim)
            ecvh = ECVH(dim, config['m_vantages'], config['raw_bits'], config['code_bits'])
            lrsq = LRSQ(dim, config['rank'], config['blocks'])
            
            bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
            index = AsteriaIndexCPU(bundle, device='cpu')
            
            # Build index
            build_start = time.time()
            db_tensor = torch.tensor(db_features, dtype=torch.float32)
            index.add(db_tensor)
            build_time = time.time() - build_start
            
            # Search
            search_start = time.time()
            query_tensor = torch.tensor(query_features, dtype=torch.float32)
            distances, indices = index.search(
                query_tensor,
                k=10,
                target_mult=config['target_mult'],
                max_radius=config['max_radius']
            )
            search_time = time.time() - search_start
            
            # Calculate metrics
            qps = len(query_features) / search_time
            
            # Calculate class-based recall (semantic recall)
            semantic_recall = self._calculate_semantic_recall(
                indices, query_labels, np.array(range(len(db_features))) % 10)
            
            # Calculate standard recall@k (using brute force ground truth)
            gt_similarities = np.dot(query_features, db_features.T)
            gt_indices = np.argpartition(-gt_similarities, 9, axis=1)[:, :10]
            standard_recall = self._calculate_standard_recall(indices, gt_indices)
            
            result = {
                'config_name': config['name'],
                'config': config,
                'build_time': build_time,
                'search_time': search_time,
                'qps': qps,
                'standard_recall@10': standard_recall,
                'semantic_recall@10': semantic_recall,
                'db_size': len(db_features),
                'query_size': len(query_features),
                'feature_dim': dim,
                'feature_model': feature_model
            }
            
            results.append(result)
            
            print(f"  QPS: {qps:.2f}")
            print(f"  Standard Recall@10: {standard_recall:.4f}")
            print(f"  Semantic Recall@10: {semantic_recall:.4f}")
            print(f"  Build time: {build_time:.2f}s")
        
        self.results['cifar10_experiment'] = results
        self._plot_cifar10_results(results)
        return results
    
    def run_scalability_study(self, 
                             max_db_size: int = 100000,
                             feature_model: str = 'resnet'):
        """Study scalability with increasing database sizes"""
        
        print("=== Running Scalability Study ===")
        
        db_sizes = [1000, 5000, 10000, 25000, 50000]
        if max_db_size > 50000:
            db_sizes.append(max_db_size)
        
        results = []
        
        for db_size in db_sizes:
            print(f"\nTesting with database size: {db_size}")
            
            # Generate or load data
            if db_size <= 10000:
                db_features, query_features, query_labels = self.setup_cifar10_experiment(
                    db_size, feature_model)
            else:
                # Use synthetic data for larger sizes
                db_features, query_features, query_labels = self._generate_synthetic_cifar10_like(db_size)
            
            # Use balanced configuration
            config = {
                'raw_bits': 32, 'code_bits': 32, 'm_vantages': 48,
                'rank': 48, 'blocks': 12, 'target_mult': 8, 'max_radius': 2
            }
            
            # Test Asteria
            dim = db_features.shape[1]
            bor = ButterflyRotation(dim)
            ecvh = ECVH(dim, config['m_vantages'], config['raw_bits'], config['code_bits'])
            lrsq = LRSQ(dim, config['rank'], config['blocks'])
            
            bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
            index = AsteriaIndexCPU(bundle, device='cpu')
            
            # Build and search
            build_start = time.time()
            db_tensor = torch.tensor(db_features, dtype=torch.float32)
            index.add(db_tensor)
            build_time = time.time() - build_start
            
            search_start = time.time()
            query_tensor = torch.tensor(query_features, dtype=torch.float32)
            distances, indices = index.search(query_tensor, k=10,
                                            target_mult=config['target_mult'],
                                            max_radius=config['max_radius'])
            search_time = time.time() - search_start
            
            qps = len(query_features) / search_time
            
            # Calculate recall
            gt_similarities = np.dot(query_features, db_features.T)
            gt_indices = np.argpartition(-gt_similarities, 9, axis=1)[:, :10]
            recall = self._calculate_standard_recall(indices, gt_indices)
            
            result = {
                'db_size': db_size,
                'build_time': build_time,
                'search_time': search_time,
                'qps': qps,
                'recall@10': recall,
                'query_size': len(query_features),
                'feature_dim': dim
            }
            
            results.append(result)
            
            print(f"  QPS: {qps:.2f}")
            print(f"  Recall@10: {recall:.4f}")
            print(f"  Build time: {build_time:.2f}s")
        
        self.results['scalability_study'] = results
        self._plot_scalability_results(results)
        return results
    
    def _calculate_standard_recall(self, retrieved_indices: np.ndarray, gt_indices: np.ndarray) -> float:
        """Calculate standard recall@k"""
        recall_sum = 0
        for i in range(len(retrieved_indices)):
            retrieved_set = set(retrieved_indices[i])
            gt_set = set(gt_indices[i])
            recall_sum += len(retrieved_set & gt_set) / len(gt_set)
        
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
    
    def _plot_cifar10_results(self, results: List[Dict]):
        """Plot CIFAR-10 experiment results"""
        
        config_names = [r['config_name'] for r in results]
        qps_values = [r['qps'] for r in results]
        standard_recall = [r['standard_recall@10'] for r in results]
        semantic_recall = [r['semantic_recall@10'] for r in results]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # QPS comparison
        bars1 = axes[0].bar(config_names, qps_values, alpha=0.7)
        axes[0].set_ylabel('Queries Per Second (QPS)')
        axes[0].set_title('Search Speed by Configuration')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, qps in zip(bars1, qps_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(qps_values)*0.01,
                        f'{qps:.1f}', ha='center', va='bottom')
        
        # Recall comparison
        x = np.arange(len(config_names))
        width = 0.35
        
        bars2 = axes[1].bar(x - width/2, standard_recall, width, label='Standard Recall@10', alpha=0.7)
        bars3 = axes[1].bar(x + width/2, semantic_recall, width, label='Semantic Recall@10', alpha=0.7)
        
        axes[1].set_ylabel('Recall@10')
        axes[1].set_title('Recall Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(config_names)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Speed vs Accuracy tradeoff
        axes[2].scatter(standard_recall, qps_values, s=100, alpha=0.7, label='Standard')
        axes[2].scatter(semantic_recall, qps_values, s=100, alpha=0.7, label='Semantic')
        
        for i, name in enumerate(config_names):
            axes[2].annotate(name, (standard_recall[i], qps_values[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[2].set_xlabel('Recall@10')
        axes[2].set_ylabel('QPS')
        axes[2].set_title('Speed vs Accuracy Tradeoff')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/cifar10_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_scalability_results(self, results: List[Dict]):
        """Plot scalability study results"""
        
        db_sizes = [r['db_size'] for r in results]
        qps_values = [r['qps'] for r in results]
        recall_values = [r['recall@10'] for r in results]
        build_times = [r['build_time'] for r in results]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # QPS vs Database Size
        axes[0].semilogx(db_sizes, qps_values, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Database Size')
        axes[0].set_ylabel('QPS')
        axes[0].set_title('Search Speed vs Database Size (Real Images)')
        axes[0].grid(True, alpha=0.3)
        
        # Recall vs Database Size
        axes[1].semilogx(db_sizes, recall_values, 'o-', linewidth=2, markersize=8, color='red')
        axes[1].set_xlabel('Database Size')
        axes[1].set_ylabel('Recall@10')
        axes[1].set_title('Recall vs Database Size (Real Images)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.1)
        
        # Build Time vs Database Size
        axes[2].loglog(db_sizes, build_times, 'o-', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Database Size')
        axes[2].set_ylabel('Build Time (seconds)')
        axes[2].set_title('Index Build Time vs Database Size')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/scalability_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str = 'real_image_results.json'):
        """Save all results"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def generate_summary(self):
        """Generate experiment summary"""
        print("\n" + "="*80)
        print("REAL IMAGE EXPERIMENT SUMMARY")
        print("="*80)
        
        for experiment_name, experiment_results in self.results.items():
            print(f"\n{experiment_name.upper()}:")
            print("-" * 40)
            
            if experiment_name == 'cifar10_experiment':
                print(f"{'Config':<12} {'QPS':<8} {'Std Recall':<12} {'Sem Recall':<12} {'Build Time':<12}")
                print("-" * 60)
                for result in experiment_results:
                    print(f"{result['config_name']:<12} {result['qps']:<8.1f} "
                          f"{result['standard_recall@10']:<12.4f} {result['semantic_recall@10']:<12.4f} "
                          f"{result['build_time']:<12.2f}")
            
            elif experiment_name == 'scalability_study':
                print(f"{'DB Size':<10} {'QPS':<8} {'Recall@10':<12} {'Build Time':<12}")
                print("-" * 45)
                for result in experiment_results:
                    print(f"{result['db_size']:<10} {result['qps']:<8.1f} "
                          f"{result['recall@10']:<12.4f} {result['build_time']:<12.2f}")

def main():
    """Run real image experiments"""
    
    experiment = RealImageExperiment()
    
    print("Starting real image experiments...")
    
    # 1. CIFAR-10 experiment
    print("\n1. Running CIFAR-10 experiment...")
    experiment.run_cifar10_experiment(subset_size=10000, feature_model='resnet')
    
    # 2. Scalability study
    print("\n2. Running scalability study...")
    experiment.run_scalability_study(max_db_size=50000)
    
    # Save results and generate summary
    experiment.save_results()
    experiment.generate_summary()
    
    print(f"\nReal image experiments completed! Results saved in '{experiment.save_dir}' directory.")

if __name__ == "__main__":
    main()
