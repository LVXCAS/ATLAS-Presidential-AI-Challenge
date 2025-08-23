"""
Distributed Training System Demo

Demonstrates the parallel training environment with:
- Multiple concurrent training jobs
- Hyperparameter optimization
- Resource-aware scheduling
- Experiment management
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any

# Add backend to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from training import (
    TrainingManager, TrainingWorker, ExperimentOptimizer, DistributedScheduler,
    TrainingConfig, OptimizationConfig, ParameterSpace, OptimizationStrategy,
    SchedulingStrategy, ResourceRequirement, ResourceMonitor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_training_system():
    """Set up the distributed training system components."""
    logger.info("=== Setting up Distributed Training System ===")
    
    # Initialize training manager
    training_manager = TrainingManager()
    await training_manager.initialize()
    await training_manager.start()
    
    # Initialize experiment optimizer
    experiment_optimizer = ExperimentOptimizer(training_manager)
    
    # Initialize distributed scheduler
    scheduler = DistributedScheduler(SchedulingStrategy.LOAD_BALANCED)
    await scheduler.initialize()
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor()
    await resource_monitor.start_monitoring()
    
    # Create worker nodes
    workers = []
    for i in range(3):
        worker_id = f"worker_{i+1}"
        capabilities = ['standard_ml', 'time_series', 'deep_learning']
        
        worker = TrainingWorker(worker_id, capabilities)
        await worker.initialize()
        await worker.start()
        workers.append(worker)
    
    logger.info(f"Created {len(workers)} training workers")
    
    return {
        'training_manager': training_manager,
        'experiment_optimizer': experiment_optimizer,
        'scheduler': scheduler,
        'resource_monitor': resource_monitor,
        'workers': workers
    }


async def demo_basic_training(training_manager: TrainingManager):
    """Demonstrate basic parallel training."""
    logger.info("=== Basic Parallel Training Demo ===")
    
    # Create multiple training configurations
    training_configs = []
    
    model_types = ['random_forest', 'xgboost', 'neural_network']
    for i, model_type in enumerate(model_types):
        config = TrainingConfig(
            model_type=model_type,
            model_name=f"{model_type}_model_{i+1}",
            parameters={
                'feature_selection': True,
                'cross_validation': True
            },
            training_data_params={
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                'lookback_days': 30,
                'size': 10000
            },
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
            hyperparameters={
                'n_estimators': 100 if model_type != 'neural_network' else None,
                'max_depth': 10 if model_type != 'neural_network' else None,
                'hidden_layers': [64, 32] if model_type == 'neural_network' else None
            }
        )
        training_configs.append(config)
    
    # Create experiment
    experiment_id = await training_manager.create_experiment(
        name="Parallel Model Comparison",
        description="Compare different model types on the same dataset",
        training_configs=training_configs,
        metadata={
            'objective': 'model_comparison',
            'dataset': 'stock_prediction',
            'created_by': 'demo_system'
        }
    )
    
    logger.info(f"Created experiment {experiment_id} with {len(training_configs)} jobs")
    
    # Start experiment
    await training_manager.start_experiment(experiment_id)
    
    # Monitor progress
    while True:
        status = await training_manager.get_experiment_status(experiment_id)
        
        logger.info(f"Experiment progress: {status['progress']:.1%} "
                   f"({status['completed_jobs']}/{status['total_jobs']} jobs)")
        
        if status['progress'] >= 1.0:
            logger.info("Experiment completed!")
            logger.info(f"Results: {status['results_summary']}")
            break
        
        await asyncio.sleep(10)
    
    return experiment_id


async def demo_hyperparameter_optimization(experiment_optimizer: ExperimentOptimizer):
    """Demonstrate hyperparameter optimization."""
    logger.info("=== Hyperparameter Optimization Demo ===")
    
    # Base configuration for optimization
    base_config = TrainingConfig(
        model_type='random_forest',
        model_name='optimized_rf_model',
        parameters={
            'feature_selection': True,
            'cross_validation': True
        },
        training_data_params={
            'symbols': ['AAPL', 'MSFT'],
            'lookback_days': 30,
            'size': 5000
        },
        epochs=30,  # Shorter for optimization demo
        batch_size=32
    )
    
    # Define parameter space to optimize
    parameter_spaces = [
        ParameterSpace(
            name='n_estimators',
            param_type='discrete',
            choices=[50, 100, 200, 300],
            dtype='int'
        ),
        ParameterSpace(
            name='max_depth',
            param_type='continuous',
            bounds=(3, 20),
            dtype='int'
        ),
        ParameterSpace(
            name='min_samples_split',
            param_type='continuous',
            bounds=(2, 20),
            dtype='int'
        ),
        ParameterSpace(
            name='min_samples_leaf',
            param_type='continuous',
            bounds=(1, 10),
            dtype='int'
        )
    ]
    
    # Optimization configuration
    opt_config = OptimizationConfig(
        strategy=OptimizationStrategy.RANDOM_SEARCH,
        parameter_spaces=parameter_spaces,
        objective_metric='accuracy',
        maximize=True,
        max_trials=20,
        max_concurrent_trials=5,
        random_seed=42
    )
    
    # Start optimization
    optimization_id = await experiment_optimizer.start_optimization(
        experiment_name="Random Forest Optimization",
        base_config=base_config,
        optimization_config=opt_config
    )
    
    logger.info(f"Started optimization {optimization_id}")
    
    # Monitor optimization progress
    while True:
        status = await experiment_optimizer.get_optimization_status(optimization_id)
        
        if status is None:
            logger.error("Optimization not found!")
            break
        
        logger.info(f"Optimization progress: {status['progress']:.1%} "
                   f"({status['completed_trials']}/{status['total_trials']} trials)")
        
        if status.get('best_trial'):
            best_trial = status['best_trial']
            logger.info(f"Current best: {best_trial['objective_value']:.4f} "
                       f"with parameters: {best_trial['parameters']}")
        
        if status['is_complete']:
            logger.info("Optimization completed!")
            
            # Get detailed results
            results = await experiment_optimizer.get_optimization_results(optimization_id)
            if results:
                logger.info(f"Best trial: {results['best_trial']['trial_id']}")
                logger.info(f"Best parameters: {results['best_trial']['parameters']}")
                logger.info(f"Best score: {results['best_trial']['objective_value']:.4f}")
                logger.info(f"Parameter importance: {results['parameter_importance']}")
            
            break
        
        await asyncio.sleep(15)
    
    return optimization_id


async def demo_resource_monitoring(scheduler: DistributedScheduler):
    """Demonstrate resource monitoring and scheduling."""
    logger.info("=== Resource Monitoring and Scheduling Demo ===")
    
    # Get scheduling metrics
    metrics = await scheduler.get_scheduling_metrics()
    
    logger.info("Scheduling Metrics:")
    logger.info(f"  Strategy: {metrics['strategy']}")
    logger.info(f"  Total jobs scheduled: {metrics['total_jobs_scheduled']}")
    logger.info(f"  Pending jobs: {metrics['pending_jobs']}")
    logger.info(f"  Active jobs: {metrics['active_jobs']}")
    logger.info(f"  Total nodes: {metrics['total_nodes']}")
    logger.info(f"  Scheduling efficiency: {metrics['scheduling_efficiency']:.2%}")
    
    # Get node status
    nodes = await scheduler.get_node_status()
    
    logger.info(f"\nNode Status ({len(nodes)} nodes):")
    for node in nodes:
        logger.info(f"  Node {node['node_id']}:")
        logger.info(f"    CPU: {node['cpu_utilization']:.1f}% "
                   f"({node['cpu_cores']} cores)")
        logger.info(f"    Memory: {node['memory_utilization']:.1f}% "
                   f"({node['memory_gb']:.1f} GB)")
        logger.info(f"    Queue length: {node['queue_length']}")
        logger.info(f"    Status: {node['status']}")


async def demo_training_strategies(training_manager: TrainingManager):
    """Demonstrate different training strategies."""
    logger.info("=== Training Strategy Comparison Demo ===")
    
    strategies = [
        ('ensemble_voting', 'Ensemble Voting'),
        ('ensemble_stacking', 'Ensemble Stacking'),
        ('transfer_learning', 'Transfer Learning')
    ]
    
    experiments = {}
    
    for strategy_type, strategy_name in strategies:
        # Create configuration for each strategy
        config = TrainingConfig(
            model_type=strategy_type,
            model_name=f"{strategy_type}_model",
            parameters={
                'base_models': ['random_forest', 'xgboost', 'neural_network'] if 'ensemble' in strategy_type else None,
                'pretrained_model': 'financial_base_model' if strategy_type == 'transfer_learning' else None
            },
            training_data_params={
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
                'lookback_days': 60,
                'size': 15000
            },
            epochs=40,
            batch_size=128,
            hyperparameters={
                'voting_method': 'soft' if strategy_type == 'ensemble_voting' else None,
                'meta_learner': 'logistic_regression' if strategy_type == 'ensemble_stacking' else None,
                'freeze_layers': 5 if strategy_type == 'transfer_learning' else None
            }
        )
        
        # Create experiment
        experiment_id = await training_manager.create_experiment(
            name=f"{strategy_name} Training",
            description=f"Training using {strategy_name} approach",
            training_configs=[config]
        )
        
        experiments[strategy_name] = experiment_id
        
        # Start experiment
        await training_manager.start_experiment(experiment_id)
        
        logger.info(f"Started {strategy_name} experiment: {experiment_id}")
    
    # Monitor all experiments
    logger.info("Monitoring all strategy experiments...")
    
    completed_experiments = set()
    
    while len(completed_experiments) < len(experiments):
        for strategy_name, experiment_id in experiments.items():
            if strategy_name in completed_experiments:
                continue
            
            status = await training_manager.get_experiment_status(experiment_id)
            
            if status['progress'] >= 1.0:
                completed_experiments.add(strategy_name)
                logger.info(f"{strategy_name} completed!")
                
                if status['results_summary']:
                    best_metric = status['results_summary'].get('best_metric', 0)
                    logger.info(f"  Best metric: {best_metric:.4f}")
            else:
                logger.info(f"{strategy_name}: {status['progress']:.1%} complete")
        
        if len(completed_experiments) < len(experiments):
            await asyncio.sleep(10)
    
    logger.info("All strategy experiments completed!")
    
    return experiments


async def cleanup_system(components):
    """Clean up system components."""
    logger.info("=== Cleaning up Training System ===")
    
    # Stop workers
    for worker in components['workers']:
        await worker.stop()
    
    # Stop resource monitor
    await components['resource_monitor'].stop_monitoring()
    
    # Stop training manager
    await components['training_manager'].stop()
    
    logger.info("Training system cleanup completed")


async def main():
    """Run the comprehensive distributed training demo."""
    logger.info("Starting Distributed Training System Demo")
    logger.info("=" * 60)
    
    try:
        # Set up training system
        components = await setup_training_system()
        
        # Wait for system to stabilize
        logger.info("Waiting for system to stabilize...")
        await asyncio.sleep(5)
        
        # Demo 1: Basic parallel training
        experiment_id = await demo_basic_training(components['training_manager'])
        
        print("\n" + "=" * 60 + "\n")
        
        # Demo 2: Hyperparameter optimization
        optimization_id = await demo_hyperparameter_optimization(components['experiment_optimizer'])
        
        print("\n" + "=" * 60 + "\n")
        
        # Demo 3: Resource monitoring
        await demo_resource_monitoring(components['scheduler'])
        
        print("\n" + "=" * 60 + "\n")
        
        # Demo 4: Training strategies
        strategy_experiments = await demo_training_strategies(components['training_manager'])
        
        # Final system metrics
        logger.info("\n=== Final System Metrics ===")
        training_metrics = await components['training_manager'].get_training_metrics()
        
        logger.info(f"Total experiments: {training_metrics['total_experiments']}")
        logger.info(f"Total jobs: {training_metrics['total_jobs']}")
        logger.info(f"Completed jobs: {training_metrics['completed_jobs']}")
        logger.info(f"Failed jobs: {training_metrics['failed_jobs']}")
        logger.info(f"Active workers: {training_metrics['active_workers']}")
        logger.info(f"System utilization: {training_metrics['system_utilization']:.2f}")
        
        logger.info("\nDemo completed successfully!")
        
        # Cleanup
        await cleanup_system(components)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())