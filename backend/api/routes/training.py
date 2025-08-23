"""
Training API routes for distributed ML training system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
import uuid

from training import (
    TrainingManager, ExperimentOptimizer, DistributedScheduler,
    TrainingConfig, OptimizationConfig, ParameterSpace, OptimizationStrategy,
    SchedulingStrategy, ResourceRequirement, ResourceMonitor
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global training system components
training_manager: Optional[TrainingManager] = None
experiment_optimizer: Optional[ExperimentOptimizer] = None
scheduler: Optional[DistributedScheduler] = None
resource_monitor: Optional[ResourceMonitor] = None


class TrainingConfigRequest(BaseModel):
    """Training configuration request model."""
    model_type: str
    model_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    training_data_params: Dict[str, Any] = Field(default_factory=dict)
    validation_split: float = Field(default=0.2, ge=0.0, le=0.8)
    epochs: int = Field(default=100, ge=1, le=10000)
    batch_size: int = Field(default=32, ge=1, le=1024)
    learning_rate: float = Field(default=0.001, ge=1e-6, le=1.0)
    early_stopping: bool = True
    early_stopping_patience: int = Field(default=10, ge=1, le=100)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class ExperimentRequest(BaseModel):
    """Experiment creation request model."""
    name: str
    description: str
    training_configs: List[TrainingConfigRequest]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OptimizationRequest(BaseModel):
    """Optimization experiment request model."""
    name: str
    base_config: TrainingConfigRequest
    strategy: str = "random_search"
    parameter_spaces: List[Dict[str, Any]]
    objective_metric: str = "accuracy"
    maximize: bool = True
    max_trials: int = Field(default=100, ge=1, le=1000)
    max_concurrent_trials: int = Field(default=10, ge=1, le=50)
    random_seed: Optional[int] = None


class ResourceRequirementRequest(BaseModel):
    """Resource requirement specification."""
    cpu_cores: float = Field(default=1.0, ge=0.1, le=64.0)
    memory_gb: float = Field(default=2.0, ge=0.1, le=512.0)
    gpu_memory_gb: float = Field(default=0.0, ge=0.0, le=80.0)
    disk_gb: float = Field(default=1.0, ge=0.1, le=1000.0)
    estimated_duration_minutes: float = Field(default=60.0, ge=1.0, le=10080.0)  # Max 1 week
    priority: int = Field(default=5, ge=1, le=10)


async def get_training_manager() -> TrainingManager:
    """Get or create training manager instance."""
    global training_manager
    if training_manager is None:
        training_manager = TrainingManager()
        await training_manager.initialize()
        await training_manager.start()
    return training_manager


async def get_experiment_optimizer() -> ExperimentOptimizer:
    """Get or create experiment optimizer instance."""
    global experiment_optimizer
    if experiment_optimizer is None:
        manager = await get_training_manager()
        experiment_optimizer = ExperimentOptimizer(manager)
    return experiment_optimizer


async def get_scheduler() -> DistributedScheduler:
    """Get or create distributed scheduler instance."""
    global scheduler
    if scheduler is None:
        scheduler = DistributedScheduler(SchedulingStrategy.LOAD_BALANCED)
        await scheduler.initialize()
    return scheduler


async def get_resource_monitor() -> ResourceMonitor:
    """Get or create resource monitor instance."""
    global resource_monitor
    if resource_monitor is None:
        resource_monitor = ResourceMonitor()
        await resource_monitor.start_monitoring()
    return resource_monitor


@router.post("/experiments")
async def create_experiment(request: ExperimentRequest) -> Dict[str, Any]:
    """Create a new training experiment."""
    try:
        manager = await get_training_manager()
        
        # Convert request models to training configs
        training_configs = []
        for config_request in request.training_configs:
            config = TrainingConfig(
                model_type=config_request.model_type,
                model_name=config_request.model_name,
                parameters=config_request.parameters,
                training_data_params=config_request.training_data_params,
                validation_split=config_request.validation_split,
                epochs=config_request.epochs,
                batch_size=config_request.batch_size,
                learning_rate=config_request.learning_rate,
                early_stopping=config_request.early_stopping,
                early_stopping_patience=config_request.early_stopping_patience,
                hyperparameters=config_request.hyperparameters
            )
            training_configs.append(config)
        
        # Create experiment
        experiment_id = await manager.create_experiment(
            name=request.name,
            description=request.description,
            training_configs=training_configs,
            metadata=request.metadata
        )
        
        return {
            'experiment_id': experiment_id,
            'status': 'created',
            'total_jobs': len(training_configs),
            'created_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str) -> Dict[str, Any]:
    """Start a training experiment."""
    try:
        manager = await get_training_manager()
        
        success = await manager.start_experiment(experiment_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
        return {
            'experiment_id': experiment_id,
            'status': 'started',
            'started_at': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments() -> List[Dict[str, Any]]:
    """List all experiments."""
    try:
        manager = await get_training_manager()
        experiments = await manager.list_experiments()
        return experiments
        
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}")
async def get_experiment_status(experiment_id: str) -> Dict[str, Any]:
    """Get experiment status and progress."""
    try:
        manager = await get_training_manager()
        
        status = await manager.get_experiment_status(experiment_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get training job status."""
    try:
        manager = await get_training_manager()
        
        status = await manager.get_job_status(job_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    """Cancel a training job."""
    try:
        manager = await get_training_manager()
        
        success = await manager.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")
        
        return {
            'job_id': job_id,
            'status': 'cancelled',
            'cancelled_at': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimization")
async def start_optimization(request: OptimizationRequest) -> Dict[str, Any]:
    """Start hyperparameter optimization."""
    try:
        optimizer = await get_experiment_optimizer()
        
        # Convert base config
        base_config = TrainingConfig(
            model_type=request.base_config.model_type,
            model_name=request.base_config.model_name,
            parameters=request.base_config.parameters,
            training_data_params=request.base_config.training_data_params,
            validation_split=request.base_config.validation_split,
            epochs=request.base_config.epochs,
            batch_size=request.base_config.batch_size,
            learning_rate=request.base_config.learning_rate,
            hyperparameters=request.base_config.hyperparameters
        )
        
        # Convert parameter spaces
        parameter_spaces = []
        for space_data in request.parameter_spaces:
            space = ParameterSpace(
                name=space_data['name'],
                param_type=space_data['param_type'],
                bounds=space_data.get('bounds'),
                choices=space_data.get('choices'),
                log_scale=space_data.get('log_scale', False),
                dtype=space_data.get('dtype', 'float')
            )
            parameter_spaces.append(space)
        
        # Create optimization config
        opt_config = OptimizationConfig(
            strategy=OptimizationStrategy(request.strategy),
            parameter_spaces=parameter_spaces,
            objective_metric=request.objective_metric,
            maximize=request.maximize,
            max_trials=request.max_trials,
            max_concurrent_trials=request.max_concurrent_trials,
            random_seed=request.random_seed
        )
        
        # Start optimization
        optimization_id = await optimizer.start_optimization(
            request.name,
            base_config,
            opt_config
        )
        
        return {
            'optimization_id': optimization_id,
            'status': 'started',
            'strategy': request.strategy,
            'max_trials': request.max_trials,
            'started_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization/{optimization_id}")
async def get_optimization_status(optimization_id: str) -> Dict[str, Any]:
    """Get optimization status."""
    try:
        optimizer = await get_experiment_optimizer()
        
        status = await optimizer.get_optimization_status(optimization_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail=f"Optimization {optimization_id} not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization/{optimization_id}/results")
async def get_optimization_results(optimization_id: str) -> Dict[str, Any]:
    """Get detailed optimization results."""
    try:
        optimizer = await get_experiment_optimizer()
        
        results = await optimizer.get_optimization_results(optimization_id)
        
        if results is None:
            raise HTTPException(status_code=404, detail=f"Optimization {optimization_id} not found")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduling/metrics")
async def get_scheduling_metrics() -> Dict[str, Any]:
    """Get scheduling system metrics."""
    try:
        scheduler_instance = await get_scheduler()
        metrics = await scheduler_instance.get_scheduling_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting scheduling metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduling/nodes")
async def get_node_status() -> List[Dict[str, Any]]:
    """Get status of all worker nodes."""
    try:
        scheduler_instance = await get_scheduler()
        nodes = await scheduler_instance.get_node_status()
        return nodes
        
    except Exception as e:
        logger.error(f"Error getting node status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources/cluster")
async def get_cluster_resources() -> Dict[str, Any]:
    """Get cluster-wide resource usage."""
    try:
        manager = await get_training_manager()
        metrics = await manager.get_training_metrics()
        
        scheduler_instance = await get_scheduler()
        scheduling_metrics = await scheduler_instance.get_scheduling_metrics()
        
        return {
            'training_metrics': metrics,
            'scheduling_metrics': scheduling_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting cluster resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workers/register")
async def register_worker(
    worker_id: str,
    capabilities: List[str] = Query(default=['standard_ml']),
    max_jobs: int = Query(default=1)
) -> Dict[str, Any]:
    """Register a new training worker."""
    try:
        # This would typically be called by worker nodes themselves
        # For now, return success status
        return {
            'worker_id': worker_id,
            'capabilities': capabilities,
            'max_concurrent_jobs': max_jobs,
            'status': 'registered',
            'registered_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error registering worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def training_health_check() -> Dict[str, Any]:
    """Health check for training system."""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check training manager
        try:
            manager = await get_training_manager()
            if manager.is_running:
                health_status['components']['training_manager'] = 'healthy'
            else:
                health_status['components']['training_manager'] = 'unhealthy'
                health_status['status'] = 'degraded'
        except Exception:
            health_status['components']['training_manager'] = 'unhealthy'
            health_status['status'] = 'degraded'
        
        # Check scheduler
        try:
            scheduler_instance = await get_scheduler()
            if scheduler_instance.scheduling_enabled:
                health_status['components']['scheduler'] = 'healthy'
            else:
                health_status['components']['scheduler'] = 'unhealthy'
                health_status['status'] = 'degraded'
        except Exception:
            health_status['components']['scheduler'] = 'unhealthy'
            health_status['status'] = 'degraded'
        
        # Check resource monitor
        try:
            monitor = await get_resource_monitor()
            if monitor.monitoring:
                health_status['components']['resource_monitor'] = 'healthy'
            else:
                health_status['components']['resource_monitor'] = 'unhealthy'
                health_status['status'] = 'degraded'
        except Exception:
            health_status['components']['resource_monitor'] = 'unhealthy'
            health_status['status'] = 'degraded'
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in training health check: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@router.get("/strategies")
async def get_optimization_strategies() -> Dict[str, Any]:
    """Get available optimization strategies."""
    return {
        'optimization_strategies': [
            {
                'name': 'grid_search',
                'display_name': 'Grid Search',
                'description': 'Exhaustive search over parameter grid'
            },
            {
                'name': 'random_search', 
                'display_name': 'Random Search',
                'description': 'Random sampling from parameter space'
            },
            {
                'name': 'bayesian_optimization',
                'display_name': 'Bayesian Optimization',
                'description': 'Gaussian process guided optimization'
            },
            {
                'name': 'genetic_algorithm',
                'display_name': 'Genetic Algorithm',
                'description': 'Evolution-based parameter optimization'
            },
            {
                'name': 'hyperband',
                'display_name': 'Hyperband',
                'description': 'Bandit-based early stopping optimization'
            }
        ],
        'scheduling_strategies': [
            {
                'name': 'fifo',
                'display_name': 'First In, First Out',
                'description': 'Process jobs in order of submission'
            },
            {
                'name': 'priority',
                'display_name': 'Priority-based',
                'description': 'Schedule based on job priority'
            },
            {
                'name': 'load_balanced',
                'display_name': 'Load Balanced',
                'description': 'Balance load across available nodes'
            },
            {
                'name': 'resource_aware',
                'display_name': 'Resource Aware',
                'description': 'Consider resource requirements for scheduling'
            },
            {
                'name': 'fair_share',
                'display_name': 'Fair Share',
                'description': 'Ensure fair resource allocation among users'
            }
        ]
    }