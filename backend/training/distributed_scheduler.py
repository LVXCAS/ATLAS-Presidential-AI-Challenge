"""
Distributed training scheduler for managing training resources and workloads across multiple nodes.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import psutil
import socket

from core.redis_manager import get_redis_manager
from .training_manager import TrainingJob, TrainingStatus, WorkerNode

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based scheduling
    LOAD_BALANCED = "load_balanced"  # Balance load across workers
    RESOURCE_AWARE = "resource_aware"  # Consider resource requirements
    FAIR_SHARE = "fair_share"  # Fair sharing among users/experiments


@dataclass
class ResourceRequirement:
    """Resource requirements for training jobs."""
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    gpu_memory_gb: float = 0.0
    disk_gb: float = 1.0
    network_bandwidth_mbps: float = 10.0
    estimated_duration_minutes: float = 60.0
    priority: int = 5  # 1-10, higher is more priority


@dataclass
class NodeResources:
    """Available resources on a worker node."""
    cpu_cores: float
    memory_gb: float
    gpu_memory_gb: float
    disk_gb: float
    network_bandwidth_mbps: float
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    disk_utilization: float = 0.0


@dataclass
class SchedulingDecision:
    """Scheduling decision for a job."""
    job_id: str
    assigned_node: Optional[str]
    estimated_start_time: datetime
    estimated_completion_time: datetime
    resource_allocation: ResourceRequirement
    priority_score: float
    rejection_reason: Optional[str] = None


class ResourceMonitor:
    """Monitor system resources on worker nodes."""
    
    def __init__(self):
        self.node_id = socket.gethostname()
        self.monitoring = False
        
    async def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        asyncio.create_task(self._monitoring_loop())
        logger.info(f"Resource monitoring started for node {self.node_id}")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        logger.info(f"Resource monitoring stopped for node {self.node_id}")
    
    async def _monitoring_loop(self):
        """Continuously monitor and report resource usage."""
        redis = get_redis_manager().client
        
        while self.monitoring:
            try:
                # Get system resources
                resources = await self._get_system_resources()
                
                # Store in Redis with TTL
                await redis.setex(
                    f"training:node:{self.node_id}:resources",
                    60,  # 1 minute TTL
                    json.dumps(resources, default=str)
                )
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_percent = memory.percent
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_gb = disk.total / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
            
            # Network information (simplified)
            network_io = psutil.net_io_counters()
            
            resources = {
                'node_id': self.node_id,
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'cores': cpu_count,
                    'utilization_percent': cpu_percent,
                    'available_cores': cpu_count * (100 - cpu_percent) / 100
                },
                'memory': {
                    'total_gb': memory_gb,
                    'utilization_percent': memory_percent,
                    'available_gb': memory_gb * (100 - memory_percent) / 100
                },
                'disk': {
                    'total_gb': disk_gb,
                    'utilization_percent': disk_percent,
                    'available_gb': disk_gb * (100 - disk_percent) / 100
                },
                'network': {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv
                },
                'gpu': {
                    'available': False,  # Would integrate with nvidia-ml-py for GPU monitoring
                    'memory_gb': 0.0,
                    'utilization_percent': 0.0
                }
            }
            
            return resources
            
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {
                'node_id': self.node_id,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }


class DistributedScheduler:
    """Intelligent scheduler for distributed training workloads."""
    
    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.LOAD_BALANCED):
        self.strategy = strategy
        self.redis = None
        self.scheduling_enabled = True
        
        # Scheduling state
        self.pending_jobs: List[Tuple[TrainingJob, ResourceRequirement]] = []
        self.scheduled_jobs: Dict[str, SchedulingDecision] = {}
        self.job_queues: Dict[str, List[str]] = {}  # Per-node job queues
        self.node_resources: Dict[str, NodeResources] = {}
        
        # Scheduling metrics
        self.total_jobs_scheduled = 0
        self.total_jobs_rejected = 0
        self.average_queue_time = 0.0
        self.resource_utilization_history = []
        
        # Fair share state (for fair share scheduling)
        self.user_usage: Dict[str, float] = {}
        self.experiment_usage: Dict[str, float] = {}
        
        logger.info(f"DistributedScheduler initialized with strategy: {strategy.value}")
    
    async def initialize(self):
        """Initialize the distributed scheduler."""
        self.redis = get_redis_manager().client
        
        # Start background tasks
        asyncio.create_task(self._scheduling_loop())
        asyncio.create_task(self._resource_monitoring_loop())
        asyncio.create_task(self._cleanup_completed_jobs_loop())
        
        logger.info("DistributedScheduler initialized")
    
    async def schedule_job(
        self,
        job: TrainingJob,
        resource_requirements: ResourceRequirement = None
    ) -> SchedulingDecision:
        """Schedule a training job on available resources."""
        
        # Default resource requirements if not specified
        if resource_requirements is None:
            resource_requirements = self._estimate_resource_requirements(job)
        
        # Add to pending jobs
        self.pending_jobs.append((job, resource_requirements))
        
        logger.info(f"Added job {job.id} to scheduling queue")
        
        # Trigger immediate scheduling attempt
        decision = await self._schedule_pending_jobs()
        
        return decision
    
    def _estimate_resource_requirements(self, job: TrainingJob) -> ResourceRequirement:
        """Estimate resource requirements based on job configuration."""
        config = job.config
        
        # Base requirements
        base_cpu = 1.0
        base_memory = 2.0
        base_duration = 60.0
        
        # Adjust based on model type
        if config.model_type == 'deep_learning':
            base_cpu = 2.0
            base_memory = 8.0
            base_duration = 180.0
        elif config.model_type == 'ensemble':
            base_cpu = 4.0
            base_memory = 4.0
            base_duration = 120.0
        
        # Adjust based on data size
        data_size = config.training_data_params.get('size', 1000)
        if data_size > 100000:
            base_memory *= 2
            base_duration *= 1.5
        elif data_size > 1000000:
            base_memory *= 4
            base_duration *= 2
        
        # Adjust based on epochs
        epochs = config.epochs
        if epochs > 100:
            base_duration *= (epochs / 100)
        
        return ResourceRequirement(
            cpu_cores=base_cpu,
            memory_gb=base_memory,
            estimated_duration_minutes=base_duration,
            priority=5  # Default priority
        )
    
    async def _scheduling_loop(self):
        """Main scheduling loop."""
        while self.scheduling_enabled:
            try:
                if self.pending_jobs:
                    await self._schedule_pending_jobs()
                
                await asyncio.sleep(10)  # Schedule every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                await asyncio.sleep(30)
    
    async def _schedule_pending_jobs(self) -> Optional[SchedulingDecision]:
        """Schedule pending jobs based on current strategy."""
        if not self.pending_jobs:
            return None
        
        # Update node resources
        await self._update_node_resources()
        
        # Sort jobs based on strategy
        sorted_jobs = self._sort_jobs_by_strategy(self.pending_jobs)
        
        scheduled_decisions = []
        
        for job, requirements in sorted_jobs[:]:  # Create a copy to iterate safely
            decision = await self._find_suitable_node(job, requirements)
            
            if decision.assigned_node:
                # Remove from pending and schedule
                self.pending_jobs.remove((job, requirements))
                self.scheduled_jobs[job.id] = decision
                
                # Add to node queue
                if decision.assigned_node not in self.job_queues:
                    self.job_queues[decision.assigned_node] = []
                self.job_queues[decision.assigned_node].append(job.id)
                
                # Update resource allocation
                await self._allocate_resources(decision.assigned_node, requirements)
                
                # Publish job assignment
                await self._publish_job_assignment(job, decision)
                
                self.total_jobs_scheduled += 1
                scheduled_decisions.append(decision)
                
                logger.info(f"Scheduled job {job.id} on node {decision.assigned_node}")
            else:
                # Job cannot be scheduled right now
                logger.debug(f"Cannot schedule job {job.id}: {decision.rejection_reason}")
        
        return scheduled_decisions[0] if scheduled_decisions else None
    
    def _sort_jobs_by_strategy(
        self, 
        jobs: List[Tuple[TrainingJob, ResourceRequirement]]
    ) -> List[Tuple[TrainingJob, ResourceRequirement]]:
        """Sort jobs based on scheduling strategy."""
        
        if self.strategy == SchedulingStrategy.FIFO:
            # Sort by creation time
            return sorted(jobs, key=lambda x: x[0].created_at)
        
        elif self.strategy == SchedulingStrategy.PRIORITY:
            # Sort by priority (higher first)
            return sorted(jobs, key=lambda x: x[1].priority, reverse=True)
        
        elif self.strategy == SchedulingStrategy.LOAD_BALANCED:
            # Sort by resource requirements (smaller jobs first for better packing)
            return sorted(jobs, key=lambda x: x[1].cpu_cores + x[1].memory_gb)
        
        elif self.strategy == SchedulingStrategy.RESOURCE_AWARE:
            # Sort by resource efficiency (duration per resource unit)
            return sorted(
                jobs, 
                key=lambda x: x[1].estimated_duration_minutes / (x[1].cpu_cores + x[1].memory_gb)
            )
        
        elif self.strategy == SchedulingStrategy.FAIR_SHARE:
            # Sort by fair share (users/experiments with less usage get priority)
            return sorted(
                jobs,
                key=lambda x: self.experiment_usage.get(x[0].experiment_id, 0.0)
            )
        
        else:
            return jobs
    
    async def _find_suitable_node(
        self, 
        job: TrainingJob, 
        requirements: ResourceRequirement
    ) -> SchedulingDecision:
        """Find a suitable node for the job."""
        
        current_time = datetime.now()
        best_node = None
        best_score = float('-inf')
        rejection_reason = "No suitable nodes available"
        
        for node_id, resources in self.node_resources.items():
            # Check if node can accommodate the job
            if not self._can_node_handle_job(resources, requirements):
                continue
            
            # Calculate scheduling score
            score = self._calculate_node_score(node_id, resources, requirements)
            
            if score > best_score:
                best_score = score
                best_node = node_id
                rejection_reason = None
        
        if best_node:
            estimated_start = current_time
            estimated_completion = estimated_start + timedelta(
                minutes=requirements.estimated_duration_minutes
            )
            
            return SchedulingDecision(
                job_id=job.id,
                assigned_node=best_node,
                estimated_start_time=estimated_start,
                estimated_completion_time=estimated_completion,
                resource_allocation=requirements,
                priority_score=best_score
            )
        else:
            self.total_jobs_rejected += 1
            return SchedulingDecision(
                job_id=job.id,
                assigned_node=None,
                estimated_start_time=current_time,
                estimated_completion_time=current_time,
                resource_allocation=requirements,
                priority_score=0.0,
                rejection_reason=rejection_reason
            )
    
    def _can_node_handle_job(
        self, 
        node_resources: NodeResources, 
        requirements: ResourceRequirement
    ) -> bool:
        """Check if a node can handle the resource requirements."""
        
        available_cpu = node_resources.cpu_cores * (1 - node_resources.cpu_utilization / 100)
        available_memory = node_resources.memory_gb * (1 - node_resources.memory_utilization / 100)
        available_gpu = node_resources.gpu_memory_gb * (1 - node_resources.gpu_utilization / 100)
        available_disk = node_resources.disk_gb * (1 - node_resources.disk_utilization / 100)
        
        return (
            available_cpu >= requirements.cpu_cores and
            available_memory >= requirements.memory_gb and
            available_gpu >= requirements.gpu_memory_gb and
            available_disk >= requirements.disk_gb
        )
    
    def _calculate_node_score(
        self,
        node_id: str,
        resources: NodeResources,
        requirements: ResourceRequirement
    ) -> float:
        """Calculate a score for assigning a job to a node."""
        
        # Base score components
        cpu_fit = (resources.cpu_cores * (1 - resources.cpu_utilization / 100)) / requirements.cpu_cores
        memory_fit = (resources.memory_gb * (1 - resources.memory_utilization / 100)) / requirements.memory_gb
        
        # Queue length penalty
        queue_length = len(self.job_queues.get(node_id, []))
        queue_penalty = 1.0 / (1.0 + queue_length * 0.1)
        
        # Resource utilization bonus (prefer nodes with moderate utilization)
        avg_utilization = (resources.cpu_utilization + resources.memory_utilization) / 2
        utilization_bonus = 1.0 - abs(avg_utilization - 70) / 100  # Prefer ~70% utilization
        
        # Combine scores
        score = (cpu_fit * 0.4 + memory_fit * 0.3 + queue_penalty * 0.2 + utilization_bonus * 0.1)
        
        return score
    
    async def _update_node_resources(self):
        """Update available node resources from Redis."""
        try:
            # Get all node resource keys
            resource_keys = await self.redis.keys("training:node:*:resources")
            
            current_nodes = set()
            
            for key in resource_keys:
                resource_data = await self.redis.get(key)
                if resource_data:
                    resources = json.loads(resource_data)
                    node_id = resources.get('node_id')
                    
                    if node_id:
                        current_nodes.add(node_id)
                        
                        self.node_resources[node_id] = NodeResources(
                            cpu_cores=resources['cpu']['cores'],
                            memory_gb=resources['memory']['total_gb'],
                            gpu_memory_gb=resources['gpu']['memory_gb'],
                            disk_gb=resources['disk']['total_gb'],
                            network_bandwidth_mbps=1000.0,  # Default
                            cpu_utilization=resources['cpu']['utilization_percent'],
                            memory_utilization=resources['memory']['utilization_percent'],
                            gpu_utilization=resources['gpu']['utilization_percent'],
                            disk_utilization=resources['disk']['utilization_percent']
                        )
            
            # Remove offline nodes
            offline_nodes = set(self.node_resources.keys()) - current_nodes
            for node_id in offline_nodes:
                if node_id in self.node_resources:
                    del self.node_resources[node_id]
                    logger.info(f"Removed offline node: {node_id}")
            
        except Exception as e:
            logger.error(f"Error updating node resources: {e}")
    
    async def _allocate_resources(self, node_id: str, requirements: ResourceRequirement):
        """Mark resources as allocated on a node."""
        # This would update the resource tracking
        # For now, we rely on the actual resource monitoring on nodes
        pass
    
    async def _publish_job_assignment(self, job: TrainingJob, decision: SchedulingDecision):
        """Publish job assignment to the assigned node."""
        assignment_data = {
            'job_id': job.id,
            'node_id': decision.assigned_node,
            'resource_allocation': {
                'cpu_cores': decision.resource_allocation.cpu_cores,
                'memory_gb': decision.resource_allocation.memory_gb,
                'gpu_memory_gb': decision.resource_allocation.gpu_memory_gb,
                'estimated_duration': decision.resource_allocation.estimated_duration_minutes
            },
            'scheduled_at': datetime.now().isoformat()
        }
        
        # Publish to node-specific channel
        await self.redis.publish(
            f"training:node:{decision.assigned_node}:assignments",
            json.dumps(assignment_data)
        )
        
        # Store assignment in Redis
        await self.redis.setex(
            f"training:assignment:{job.id}",
            3600,  # 1 hour TTL
            json.dumps(assignment_data)
        )
    
    async def _resource_monitoring_loop(self):
        """Monitor overall resource usage across the cluster."""
        while self.scheduling_enabled:
            try:
                # Calculate cluster utilization
                if self.node_resources:
                    total_cpu = sum(n.cpu_cores for n in self.node_resources.values())
                    total_memory = sum(n.memory_gb for n in self.node_resources.values())
                    
                    used_cpu = sum(n.cpu_cores * n.cpu_utilization / 100 for n in self.node_resources.values())
                    used_memory = sum(n.memory_gb * n.memory_utilization / 100 for n in self.node_resources.values())
                    
                    cluster_utilization = {
                        'timestamp': datetime.now().isoformat(),
                        'cpu_utilization': (used_cpu / total_cpu) * 100 if total_cpu > 0 else 0,
                        'memory_utilization': (used_memory / total_memory) * 100 if total_memory > 0 else 0,
                        'total_nodes': len(self.node_resources),
                        'active_jobs': sum(len(queue) for queue in self.job_queues.values()),
                        'pending_jobs': len(self.pending_jobs)
                    }
                    
                    self.resource_utilization_history.append(cluster_utilization)
                    
                    # Keep only last 100 entries
                    if len(self.resource_utilization_history) > 100:
                        self.resource_utilization_history = self.resource_utilization_history[-100:]
                    
                    # Store in Redis
                    await self.redis.setex(
                        "training:cluster:utilization",
                        300,  # 5 minute TTL
                        json.dumps(cluster_utilization)
                    )
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_completed_jobs_loop(self):
        """Clean up completed job assignments."""
        while self.scheduling_enabled:
            try:
                # Get completed jobs
                completed_jobs = []
                
                for job_id, decision in list(self.scheduled_jobs.items()):
                    # Check if job is completed (would query training manager)
                    # For now, simulate based on estimated completion time
                    if datetime.now() > decision.estimated_completion_time:
                        completed_jobs.append(job_id)
                
                # Clean up completed jobs
                for job_id in completed_jobs:
                    if job_id in self.scheduled_jobs:
                        decision = self.scheduled_jobs[job_id]
                        
                        # Remove from node queue
                        if decision.assigned_node in self.job_queues:
                            if job_id in self.job_queues[decision.assigned_node]:
                                self.job_queues[decision.assigned_node].remove(job_id)
                        
                        # Remove from scheduled jobs
                        del self.scheduled_jobs[job_id]
                        
                        logger.debug(f"Cleaned up completed job {job_id}")
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in job cleanup: {e}")
                await asyncio.sleep(300)
    
    async def get_scheduling_metrics(self) -> Dict[str, Any]:
        """Get scheduling performance metrics."""
        
        # Calculate average queue time
        if self.pending_jobs:
            current_time = datetime.now()
            queue_times = [(current_time - job.created_at).total_seconds() / 60 
                          for job, _ in self.pending_jobs]
            avg_queue_time = sum(queue_times) / len(queue_times)
        else:
            avg_queue_time = 0.0
        
        # Resource utilization
        cluster_util = {}
        if self.node_resources:
            total_cpu = sum(n.cpu_cores for n in self.node_resources.values())
            total_memory = sum(n.memory_gb for n in self.node_resources.values())
            
            used_cpu = sum(n.cpu_cores * n.cpu_utilization / 100 for n in self.node_resources.values())
            used_memory = sum(n.memory_gb * n.memory_utilization / 100 for n in self.node_resources.values())
            
            cluster_util = {
                'cpu_utilization': (used_cpu / total_cpu) * 100 if total_cpu > 0 else 0,
                'memory_utilization': (used_memory / total_memory) * 100 if total_memory > 0 else 0
            }
        
        return {
            'strategy': self.strategy.value,
            'total_jobs_scheduled': self.total_jobs_scheduled,
            'total_jobs_rejected': self.total_jobs_rejected,
            'pending_jobs': len(self.pending_jobs),
            'active_jobs': sum(len(queue) for queue in self.job_queues.values()),
            'average_queue_time_minutes': avg_queue_time,
            'total_nodes': len(self.node_resources),
            'cluster_utilization': cluster_util,
            'scheduling_efficiency': (self.total_jobs_scheduled / 
                                    (self.total_jobs_scheduled + self.total_jobs_rejected)) 
                                   if (self.total_jobs_scheduled + self.total_jobs_rejected) > 0 else 1.0
        }
    
    async def get_node_status(self) -> List[Dict[str, Any]]:
        """Get status of all nodes."""
        nodes_status = []
        
        for node_id, resources in self.node_resources.items():
            queue_length = len(self.job_queues.get(node_id, []))
            
            nodes_status.append({
                'node_id': node_id,
                'cpu_cores': resources.cpu_cores,
                'cpu_utilization': resources.cpu_utilization,
                'memory_gb': resources.memory_gb,
                'memory_utilization': resources.memory_utilization,
                'gpu_memory_gb': resources.gpu_memory_gb,
                'gpu_utilization': resources.gpu_utilization,
                'queue_length': queue_length,
                'status': 'online' if datetime.now() else 'offline'
            })
        
        return nodes_status