"""
Training Manager for distributed ML model training and experiment management.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import pickle
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import threading

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from core.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """Training configuration for a model."""
    model_type: str
    model_name: str
    parameters: Dict[str, Any]
    training_data_params: Dict[str, Any]
    validation_split: float = 0.2
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping: bool = True
    early_stopping_patience: int = 10
    checkpointing: bool = True
    checkpoint_frequency: int = 10
    metrics_to_track: List[str] = field(default_factory=lambda: ['loss', 'accuracy'])
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """Training job representation."""
    id: str
    experiment_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    progress: float = 0.0
    current_epoch: int = 0
    best_metric: Optional[float] = None
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    model_path: Optional[str] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class Experiment:
    """Training experiment containing multiple jobs."""
    id: str
    name: str
    description: str
    status: ExperimentStatus = ExperimentStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    jobs: List[TrainingJob] = field(default_factory=list)
    results_summary: Dict[str, Any] = field(default_factory=dict)
    best_job_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerNode:
    """Distributed training worker node."""
    id: str
    host: str
    port: int
    status: str = "idle"  # idle, busy, offline
    capabilities: List[str] = field(default_factory=list)
    current_jobs: List[str] = field(default_factory=list)
    max_concurrent_jobs: int = 1
    last_heartbeat: datetime = field(default_factory=datetime.now)
    resource_usage: Dict[str, Any] = field(default_factory=dict)


class TrainingWorker:
    """Individual training worker that executes training jobs."""
    
    def __init__(self, worker_id: str, capabilities: List[str] = None):
        self.worker_id = worker_id
        self.capabilities = capabilities or ['standard_ml', 'time_series', 'deep_learning']
        self.current_jobs: Dict[str, TrainingJob] = {}
        self.is_running = False
        self.redis = None
        
        # Worker statistics
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.total_training_time = 0.0
        
        logger.info(f"TrainingWorker {worker_id} initialized with capabilities: {capabilities}")
    
    async def initialize(self):
        """Initialize worker resources."""
        self.redis = get_redis_manager().client
        self.is_running = True
        logger.info(f"TrainingWorker {self.worker_id} initialized")
    
    async def start(self):
        """Start the training worker."""
        if not self.is_running:
            await self.initialize()
        
        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())
        
        # Start job processing loop
        asyncio.create_task(self._process_jobs_loop())
        
        logger.info(f"TrainingWorker {self.worker_id} started")
    
    async def stop(self):
        """Stop the training worker."""
        self.is_running = False
        
        # Cancel running jobs
        for job_id in list(self.current_jobs.keys()):
            await self.cancel_job(job_id)
        
        logger.info(f"TrainingWorker {self.worker_id} stopped")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat to training manager."""
        while self.is_running:
            try:
                heartbeat_data = {
                    'worker_id': self.worker_id,
                    'status': 'busy' if self.current_jobs else 'idle',
                    'timestamp': datetime.now().isoformat(),
                    'current_jobs': list(self.current_jobs.keys()),
                    'jobs_completed': self.jobs_completed,
                    'jobs_failed': self.jobs_failed,
                    'capabilities': self.capabilities
                }
                
                await self.redis.setex(
                    f"training:worker:{self.worker_id}:heartbeat",
                    30,  # 30 second TTL
                    json.dumps(heartbeat_data)
                )
                
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_jobs_loop(self):
        """Process training jobs from the queue."""
        while self.is_running:
            try:
                # Check for available job slots
                if len(self.current_jobs) < 1:  # Single job per worker for now
                    # Try to get a job from queue
                    job_data = await self.redis.blpop(
                        f"training:queue:{self.worker_id}",
                        timeout=5
                    )
                    
                    if job_data:
                        queue_name, job_json = job_data
                        job_dict = json.loads(job_json)
                        
                        # Create training job object
                        job = TrainingJob(**job_dict)
                        
                        # Start training in background
                        asyncio.create_task(self._execute_training_job(job))
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in job processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _execute_training_job(self, job: TrainingJob):
        """Execute a training job."""
        job_id = job.id
        
        try:
            logger.info(f"Starting training job {job_id}")
            
            # Update job status
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            job.worker_id = self.worker_id
            self.current_jobs[job_id] = job
            
            # Save job status to Redis
            await self._save_job_status(job)
            
            # Execute the actual training
            await self._run_training(job)
            
            # Mark job as completed
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0
            
            self.jobs_completed += 1
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            
            self.jobs_failed += 1
        
        finally:
            # Clean up
            if job_id in self.current_jobs:
                del self.current_jobs[job_id]
            
            # Save final job status
            await self._save_job_status(job)
    
    async def _run_training(self, job: TrainingJob):
        """Run the actual model training."""
        config = job.config
        
        # Simulate training process
        logger.info(f"Training {config.model_type} model: {config.model_name}")
        
        # Create mock training data
        training_data = await self._prepare_training_data(config)
        
        # Initialize model (mock)
        model = await self._initialize_model(config)
        
        # Training loop
        for epoch in range(config.epochs):
            if job.status == TrainingStatus.CANCELLED:
                break
            
            # Simulate training step
            await self._training_step(job, model, training_data, epoch)
            
            # Update progress
            job.current_epoch = epoch + 1
            job.progress = (epoch + 1) / config.epochs
            
            # Save progress periodically
            if epoch % 10 == 0:
                await self._save_job_status(job)
            
            # Simulate training time
            await asyncio.sleep(0.1)  # Quick simulation
        
        # Save final model
        model_path = await self._save_model(job, model)
        job.model_path = model_path
    
    async def _prepare_training_data(self, config: TrainingConfig) -> Dict[str, Any]:
        """Prepare training data based on configuration."""
        # Mock data preparation
        if NUMPY_AVAILABLE:
            data_size = config.training_data_params.get('size', 1000)
            feature_dim = config.training_data_params.get('feature_dim', 10)
            
            X = np.random.randn(data_size, feature_dim)
            y = np.random.randint(0, 2, size=data_size)  # Binary classification
            
            return {'X': X, 'y': y}
        else:
            # Fallback without numpy
            return {'data': 'mock_data'}
    
    async def _initialize_model(self, config: TrainingConfig) -> Dict[str, Any]:
        """Initialize the ML model."""
        # Mock model initialization
        model = {
            'type': config.model_type,
            'name': config.model_name,
            'parameters': config.parameters,
            'hyperparameters': config.hyperparameters,
            'weights': None,  # Would contain actual model weights
            'architecture': 'mock_architecture'
        }
        
        return model
    
    async def _training_step(self, job: TrainingJob, model: Dict[str, Any], data: Dict[str, Any], epoch: int):
        """Execute one training step."""
        # Simulate training metrics
        if NUMPY_AVAILABLE:
            loss = 1.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.1)
            accuracy = 1.0 - np.exp(-epoch * 0.05) + np.random.normal(0, 0.05)
        else:
            loss = 1.0 - epoch * 0.01 + (hash(str(epoch)) % 100) * 0.001
            accuracy = 0.5 + epoch * 0.005 + (hash(str(epoch + 1)) % 100) * 0.001
        
        # Ensure metrics are in reasonable ranges
        loss = max(0.01, min(2.0, loss))
        accuracy = max(0.0, min(1.0, accuracy))
        
        # Track metrics
        metrics = {
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        job.metrics_history.append(metrics)
        
        # Track best metric
        if job.best_metric is None or accuracy > job.best_metric:
            job.best_metric = accuracy
        
        # Add log entry
        job.logs.append({
            'level': 'info',
            'message': f'Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}',
            'timestamp': datetime.now().isoformat()
        })
    
    async def _save_model(self, job: TrainingJob, model: Dict[str, Any]) -> str:
        """Save the trained model."""
        models_dir = Path("models") / "trained"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{job.config.model_name}_{job.id}.pkl"
        model_path = models_dir / model_filename
        
        # Save model (mock)
        model_data = {
            'model': model,
            'config': asdict(job.config),
            'training_metrics': job.metrics_history,
            'best_metric': job.best_metric,
            'job_id': job.id,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    async def _save_job_status(self, job: TrainingJob):
        """Save job status to Redis."""
        job_data = asdict(job)
        
        # Convert datetime objects to ISO strings for JSON serialization
        for key, value in job_data.items():
            if isinstance(value, datetime):
                job_data[key] = value.isoformat() if value else None
        
        await self.redis.setex(
            f"training:job:{job.id}",
            3600,  # 1 hour TTL
            json.dumps(job_data, default=str)
        )
    
    async def cancel_job(self, job_id: str):
        """Cancel a running training job."""
        if job_id in self.current_jobs:
            job = self.current_jobs[job_id]
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            
            await self._save_job_status(job)
            
            logger.info(f"Training job {job_id} cancelled")


class TrainingManager:
    """Manages distributed training jobs and experiments."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # State management
        self.experiments: Dict[str, Experiment] = {}
        self.jobs: Dict[str, TrainingJob] = {}
        self.workers: Dict[str, WorkerNode] = {}
        
        # Redis connection
        self.redis = None
        
        # Resource management
        self.max_concurrent_jobs = self.config.get('max_concurrent_jobs', 10)
        self.job_timeout = self.config.get('job_timeout', 3600)  # 1 hour
        
        # Executors for parallel processing
        self.process_executor = None
        self.thread_executor = None
        
        self.is_running = False
        
        logger.info("TrainingManager initialized")
    
    async def initialize(self):
        """Initialize the training manager."""
        self.redis = get_redis_manager().client
        
        # Initialize executors
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers//2)
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Load existing experiments and jobs from Redis
        await self._load_persistent_state()
        
        self.is_running = True
        
        logger.info("TrainingManager initialized")
    
    async def start(self):
        """Start the training manager."""
        if not self.is_running:
            await self.initialize()
        
        # Start background tasks
        asyncio.create_task(self._monitor_jobs())
        asyncio.create_task(self._monitor_workers())
        asyncio.create_task(self._cleanup_completed_jobs())
        
        logger.info("TrainingManager started")
    
    async def stop(self):
        """Stop the training manager."""
        self.is_running = False
        
        # Cancel all running jobs
        for job_id in list(self.jobs.keys()):
            if self.jobs[job_id].status == TrainingStatus.RUNNING:
                await self.cancel_job(job_id)
        
        # Shutdown executors
        if self.process_executor:
            self.process_executor.shutdown(wait=False)
        if self.thread_executor:
            self.thread_executor.shutdown(wait=False)
        
        logger.info("TrainingManager stopped")
    
    async def create_experiment(
        self,
        name: str,
        description: str,
        training_configs: List[TrainingConfig],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a new training experiment."""
        experiment_id = str(uuid.uuid4())
        
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            metadata=metadata or {}
        )
        
        # Create training jobs for each configuration
        for config in training_configs:
            job = TrainingJob(
                id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                config=config
            )
            
            experiment.jobs.append(job)
            self.jobs[job.id] = job
        
        self.experiments[experiment_id] = experiment
        
        # Save to Redis
        await self._save_experiment(experiment)
        
        logger.info(f"Created experiment {experiment_id} with {len(training_configs)} jobs")
        
        return experiment_id
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start running an experiment."""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()
        
        # Queue all jobs for execution
        for job in experiment.jobs:
            await self._queue_job(job)
        
        await self._save_experiment(experiment)
        
        logger.info(f"Started experiment {experiment_id}")
        return True
    
    async def _queue_job(self, job: TrainingJob):
        """Queue a job for execution."""
        job.status = TrainingStatus.QUEUED
        
        # Find available worker or use default queue
        worker_id = await self._find_available_worker(job.config)
        
        if not worker_id:
            worker_id = "default"
        
        # Serialize job and add to queue
        job_data = asdict(job)
        
        # Convert datetime objects
        for key, value in job_data.items():
            if isinstance(value, datetime):
                job_data[key] = value.isoformat() if value else None
        
        await self.redis.lpush(
            f"training:queue:{worker_id}",
            json.dumps(job_data, default=str)
        )
        
        logger.info(f"Queued job {job.id} for worker {worker_id}")
    
    async def _find_available_worker(self, config: TrainingConfig) -> Optional[str]:
        """Find an available worker for the given configuration."""
        # Check worker heartbeats
        worker_keys = await self.redis.keys("training:worker:*:heartbeat")
        
        available_workers = []
        
        for key in worker_keys:
            worker_data = await self.redis.get(key)
            if worker_data:
                worker_info = json.loads(worker_data)
                
                # Check if worker is idle and has required capabilities
                if (worker_info['status'] == 'idle' and 
                    config.model_type in worker_info.get('capabilities', [])):
                    available_workers.append(worker_info['worker_id'])
        
        return available_workers[0] if available_workers else None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
            return False
        
        job.status = TrainingStatus.CANCELLED
        job.completed_at = datetime.now()
        
        # Notify worker if job is running
        if job.worker_id:
            await self.redis.publish(
                f"training:worker:{job.worker_id}:cancel",
                job_id
            )
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    async def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment status and progress."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        # Calculate progress
        total_jobs = len(experiment.jobs)
        completed_jobs = sum(1 for job in experiment.jobs 
                           if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED])
        progress = completed_jobs / total_jobs if total_jobs > 0 else 0
        
        # Check if experiment is complete
        if completed_jobs == total_jobs and experiment.status == ExperimentStatus.RUNNING:
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.now()
            
            # Find best job
            best_job = max(
                [job for job in experiment.jobs if job.status == TrainingStatus.COMPLETED],
                key=lambda j: j.best_metric or 0,
                default=None
            )
            
            if best_job:
                experiment.best_job_id = best_job.id
                experiment.results_summary = {
                    'best_metric': best_job.best_metric,
                    'best_job_id': best_job.id,
                    'total_jobs': total_jobs,
                    'successful_jobs': sum(1 for job in experiment.jobs 
                                         if job.status == TrainingStatus.COMPLETED),
                    'failed_jobs': sum(1 for job in experiment.jobs 
                                     if job.status == TrainingStatus.FAILED)
                }
        
        return {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': experiment.status.value,
            'progress': progress,
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'results_summary': experiment.results_summary,
            'created_at': experiment.created_at.isoformat(),
            'started_at': experiment.started_at.isoformat() if experiment.started_at else None,
            'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None
        }
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and metrics."""
        if job_id not in self.jobs:
            # Try to load from Redis
            job_data = await self.redis.get(f"training:job:{job_id}")
            if not job_data:
                return None
            
            job_dict = json.loads(job_data)
            # Convert datetime strings back
            for key, value in job_dict.items():
                if key.endswith('_at') and value:
                    job_dict[key] = datetime.fromisoformat(value)
            
            return job_dict
        
        job = self.jobs[job_id]
        job_dict = asdict(job)
        
        # Convert datetime objects for JSON serialization
        for key, value in job_dict.items():
            if isinstance(value, datetime):
                job_dict[key] = value.isoformat() if value else None
        
        return job_dict
    
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        experiments_list = []
        
        for experiment in self.experiments.values():
            experiment_info = {
                'id': experiment.id,
                'name': experiment.name,
                'description': experiment.description,
                'status': experiment.status.value,
                'total_jobs': len(experiment.jobs),
                'created_at': experiment.created_at.isoformat(),
                'started_at': experiment.started_at.isoformat() if experiment.started_at else None,
                'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None
            }
            
            experiments_list.append(experiment_info)
        
        return experiments_list
    
    async def _monitor_jobs(self):
        """Monitor job timeouts and status updates."""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check for job timeouts
                for job in list(self.jobs.values()):
                    if (job.status == TrainingStatus.RUNNING and 
                        job.started_at and
                        (current_time - job.started_at).seconds > self.job_timeout):
                        
                        logger.warning(f"Job {job.id} timed out")
                        await self.cancel_job(job.id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in job monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_workers(self):
        """Monitor worker health and availability."""
        while self.is_running:
            try:
                # Get all worker heartbeats
                worker_keys = await self.redis.keys("training:worker:*:heartbeat")
                
                active_workers = set()
                
                for key in worker_keys:
                    worker_data = await self.redis.get(key)
                    if worker_data:
                        worker_info = json.loads(worker_data)
                        worker_id = worker_info['worker_id']
                        active_workers.add(worker_id)
                        
                        # Update worker info
                        if worker_id not in self.workers:
                            self.workers[worker_id] = WorkerNode(
                                id=worker_id,
                                host="localhost",  # Would be actual host
                                port=0,  # Would be actual port
                                capabilities=worker_info.get('capabilities', [])
                            )
                        
                        self.workers[worker_id].status = worker_info['status']
                        self.workers[worker_id].current_jobs = worker_info['current_jobs']
                        self.workers[worker_id].last_heartbeat = datetime.fromisoformat(
                            worker_info['timestamp']
                        )
                
                # Remove offline workers
                offline_workers = set(self.workers.keys()) - active_workers
                for worker_id in offline_workers:
                    if worker_id in self.workers:
                        self.workers[worker_id].status = "offline"
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in worker monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_completed_jobs(self):
        """Clean up completed jobs from memory."""
        while self.is_running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=1)  # Keep for 1 hour
                
                jobs_to_remove = []
                
                for job_id, job in self.jobs.items():
                    if (job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED] and
                        job.completed_at and job.completed_at < cutoff_time):
                        jobs_to_remove.append(job_id)
                
                for job_id in jobs_to_remove:
                    del self.jobs[job_id]
                    logger.debug(f"Cleaned up job {job_id}")
                
                await asyncio.sleep(1800)  # Clean up every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in job cleanup: {e}")
                await asyncio.sleep(600)
    
    async def _load_persistent_state(self):
        """Load experiments and jobs from Redis."""
        try:
            # Load experiments
            experiment_keys = await self.redis.keys("training:experiment:*")
            for key in experiment_keys:
                data = await self.redis.get(key)
                if data:
                    experiment_dict = json.loads(data)
                    # Convert datetime strings back
                    for dt_key in ['created_at', 'started_at', 'completed_at']:
                        if experiment_dict.get(dt_key):
                            experiment_dict[dt_key] = datetime.fromisoformat(experiment_dict[dt_key])
                    
                    experiment = Experiment(**experiment_dict)
                    self.experiments[experiment.id] = experiment
            
            logger.info(f"Loaded {len(self.experiments)} experiments from Redis")
            
        except Exception as e:
            logger.error(f"Error loading persistent state: {e}")
    
    async def _save_experiment(self, experiment: Experiment):
        """Save experiment to Redis."""
        experiment_dict = asdict(experiment)
        
        # Convert datetime objects
        for key, value in experiment_dict.items():
            if isinstance(value, datetime):
                experiment_dict[key] = value.isoformat() if value else None
        
        await self.redis.setex(
            f"training:experiment:{experiment.id}",
            86400,  # 24 hour TTL
            json.dumps(experiment_dict, default=str)
        )
    
    async def get_training_metrics(self) -> Dict[str, Any]:
        """Get overall training system metrics."""
        total_jobs = len(self.jobs)
        running_jobs = sum(1 for job in self.jobs.values() if job.status == TrainingStatus.RUNNING)
        completed_jobs = sum(1 for job in self.jobs.values() if job.status == TrainingStatus.COMPLETED)
        failed_jobs = sum(1 for job in self.jobs.values() if job.status == TrainingStatus.FAILED)
        
        active_workers = sum(1 for worker in self.workers.values() if worker.status != "offline")
        
        return {
            'total_experiments': len(self.experiments),
            'total_jobs': total_jobs,
            'running_jobs': running_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'active_workers': active_workers,
            'total_workers': len(self.workers),
            'system_utilization': running_jobs / max(active_workers, 1),
            'timestamp': datetime.now().isoformat()
        }