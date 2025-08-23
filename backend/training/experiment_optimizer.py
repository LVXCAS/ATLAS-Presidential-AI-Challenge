"""
Hyperparameter optimization and automated experiment design for ML training.
"""

import asyncio
import logging
import random
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .training_manager import TrainingConfig, TrainingManager

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    HYPERBAND = "hyperband"


@dataclass
class ParameterSpace:
    """Definition of hyperparameter search space."""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Optional[Tuple[float, float]] = None  # For continuous parameters
    choices: Optional[List[Any]] = None  # For discrete/categorical parameters
    log_scale: bool = False  # Use log scale for continuous parameters
    dtype: str = 'float'  # 'float', 'int', 'str'


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    strategy: OptimizationStrategy
    parameter_spaces: List[ParameterSpace]
    objective_metric: str = 'accuracy'  # Metric to optimize
    maximize: bool = True  # Whether to maximize or minimize the objective
    max_trials: int = 100
    max_concurrent_trials: int = 10
    early_stopping: bool = True
    early_stopping_patience: int = 20
    budget_type: str = 'iterations'  # 'iterations', 'time', 'resources'
    total_budget: float = 100.0
    min_budget: float = 1.0
    max_budget: float = 100.0
    random_seed: Optional[int] = None


@dataclass
class Trial:
    """Individual optimization trial."""
    id: str
    parameters: Dict[str, Any]
    budget: float
    objective_value: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = 'pending'  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    training_job_id: Optional[str] = None


class ExperimentOptimizer:
    """Automated hyperparameter optimization and experiment design."""
    
    def __init__(self, training_manager: TrainingManager):
        self.training_manager = training_manager
        self.optimization_history: Dict[str, List[Trial]] = {}
        self.current_optimizations: Dict[str, OptimizationConfig] = {}
        
        # Bayesian optimization state (simplified)
        self.gp_models: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ExperimentOptimizer initialized")
    
    async def start_optimization(
        self,
        experiment_name: str,
        base_config: TrainingConfig,
        optimization_config: OptimizationConfig
    ) -> str:
        """Start a hyperparameter optimization experiment."""
        optimization_id = str(uuid.uuid4())
        
        self.current_optimizations[optimization_id] = optimization_config
        self.optimization_history[optimization_id] = []
        
        # Generate trials based on strategy
        trials = await self._generate_trials(optimization_config)
        
        logger.info(f"Starting optimization {optimization_id} with {len(trials)} trials")
        
        # Start optimization in background
        asyncio.create_task(self._run_optimization(
            optimization_id, experiment_name, base_config, optimization_config, trials
        ))
        
        return optimization_id
    
    async def _generate_trials(self, config: OptimizationConfig) -> List[Trial]:
        """Generate trials based on optimization strategy."""
        trials = []
        
        if config.strategy == OptimizationStrategy.GRID_SEARCH:
            trials = await self._generate_grid_search_trials(config)
        elif config.strategy == OptimizationStrategy.RANDOM_SEARCH:
            trials = await self._generate_random_search_trials(config)
        elif config.strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
            trials = await self._generate_bayesian_trials(config)
        elif config.strategy == OptimizationStrategy.GENETIC_ALGORITHM:
            trials = await self._generate_genetic_trials(config)
        elif config.strategy == OptimizationStrategy.HYPERBAND:
            trials = await self._generate_hyperband_trials(config)
        else:
            logger.warning(f"Unknown optimization strategy: {config.strategy}")
            trials = await self._generate_random_search_trials(config)
        
        return trials
    
    async def _generate_grid_search_trials(self, config: OptimizationConfig) -> List[Trial]:
        """Generate trials for grid search."""
        trials = []
        
        # Create parameter grids
        parameter_grids = []
        for param_space in config.parameter_spaces:
            if param_space.param_type == 'continuous':
                # For continuous parameters, create discrete grid
                if param_space.bounds:
                    min_val, max_val = param_space.bounds
                    grid_size = min(10, int(math.sqrt(config.max_trials)))  # Reasonable grid size
                    
                    if param_space.log_scale:
                        values = np.logspace(np.log10(min_val), np.log10(max_val), grid_size) if NUMPY_AVAILABLE else [min_val + i * (max_val - min_val) / (grid_size - 1) for i in range(grid_size)]
                    else:
                        values = np.linspace(min_val, max_val, grid_size) if NUMPY_AVAILABLE else [min_val + i * (max_val - min_val) / (grid_size - 1) for i in range(grid_size)]
                    
                    if param_space.dtype == 'int':
                        values = [int(v) for v in values]
                    
                    parameter_grids.append((param_space.name, list(values)))
            elif param_space.param_type in ['discrete', 'categorical']:
                if param_space.choices:
                    parameter_grids.append((param_space.name, param_space.choices))
        
        # Generate all combinations
        from itertools import product
        
        param_names = [name for name, _ in parameter_grids]
        param_values = [values for _, values in parameter_grids]
        
        combinations = list(product(*param_values))[:config.max_trials]
        
        for i, combo in enumerate(combinations):
            parameters = dict(zip(param_names, combo))
            
            trial = Trial(
                id=str(uuid.uuid4()),
                parameters=parameters,
                budget=config.max_budget
            )
            trials.append(trial)
        
        return trials
    
    async def _generate_random_search_trials(self, config: OptimizationConfig) -> List[Trial]:
        """Generate trials for random search."""
        trials = []
        
        if config.random_seed:
            random.seed(config.random_seed)
            if NUMPY_AVAILABLE:
                np.random.seed(config.random_seed)
        
        for i in range(config.max_trials):
            parameters = {}
            
            for param_space in config.parameter_spaces:
                if param_space.param_type == 'continuous' and param_space.bounds:
                    min_val, max_val = param_space.bounds
                    
                    if param_space.log_scale:
                        if NUMPY_AVAILABLE:
                            value = np.random.uniform(np.log10(min_val), np.log10(max_val))
                            value = 10 ** value
                        else:
                            value = min_val * ((max_val / min_val) ** random.random())
                    else:
                        value = random.uniform(min_val, max_val) if not NUMPY_AVAILABLE else np.random.uniform(min_val, max_val)
                    
                    if param_space.dtype == 'int':
                        value = int(value)
                    
                    parameters[param_space.name] = value
                    
                elif param_space.param_type in ['discrete', 'categorical'] and param_space.choices:
                    parameters[param_space.name] = random.choice(param_space.choices)
            
            trial = Trial(
                id=str(uuid.uuid4()),
                parameters=parameters,
                budget=config.max_budget
            )
            trials.append(trial)
        
        return trials
    
    async def _generate_bayesian_trials(self, config: OptimizationConfig) -> List[Trial]:
        """Generate trials for Bayesian optimization (simplified implementation)."""
        trials = []
        
        # Start with random trials
        n_random = min(10, config.max_trials // 4)
        random_config = OptimizationConfig(
            strategy=OptimizationStrategy.RANDOM_SEARCH,
            parameter_spaces=config.parameter_spaces,
            max_trials=n_random
        )
        
        random_trials = await self._generate_random_search_trials(random_config)
        trials.extend(random_trials)
        
        # For subsequent trials, we would use acquisition functions
        # This is a simplified version that generates remaining trials randomly
        remaining_trials = config.max_trials - n_random
        if remaining_trials > 0:
            remaining_config = OptimizationConfig(
                strategy=OptimizationStrategy.RANDOM_SEARCH,
                parameter_spaces=config.parameter_spaces,
                max_trials=remaining_trials
            )
            more_trials = await self._generate_random_search_trials(remaining_config)
            trials.extend(more_trials)
        
        return trials
    
    async def _generate_genetic_trials(self, config: OptimizationConfig) -> List[Trial]:
        """Generate trials for genetic algorithm."""
        trials = []
        
        # Start with random population
        population_size = min(20, config.max_trials // 3)
        generations = config.max_trials // population_size
        
        # Initial population
        initial_config = OptimizationConfig(
            strategy=OptimizationStrategy.RANDOM_SEARCH,
            parameter_spaces=config.parameter_spaces,
            max_trials=population_size
        )
        
        population = await self._generate_random_search_trials(initial_config)
        trials.extend(population)
        
        # Generate offspring for subsequent generations
        for gen in range(generations - 1):
            offspring = []
            
            for _ in range(population_size):
                # Simple mutation of existing parameters
                if population:
                    parent = random.choice(population)
                    child_params = parent.parameters.copy()
                    
                    # Mutate one parameter
                    param_space = random.choice(config.parameter_spaces)
                    
                    if param_space.param_type == 'continuous' and param_space.bounds:
                        current_val = child_params.get(param_space.name, 0)
                        min_val, max_val = param_space.bounds
                        
                        # Add Gaussian noise
                        mutation_rate = 0.1
                        noise = random.gauss(0, (max_val - min_val) * mutation_rate)
                        new_val = max(min_val, min(max_val, current_val + noise))
                        
                        if param_space.dtype == 'int':
                            new_val = int(new_val)
                        
                        child_params[param_space.name] = new_val
                    
                    elif param_space.param_type in ['discrete', 'categorical'] and param_space.choices:
                        child_params[param_space.name] = random.choice(param_space.choices)
                    
                    offspring.append(Trial(
                        id=str(uuid.uuid4()),
                        parameters=child_params,
                        budget=config.max_budget
                    ))
            
            trials.extend(offspring)
            population = offspring
        
        return trials[:config.max_trials]
    
    async def _generate_hyperband_trials(self, config: OptimizationConfig) -> List[Trial]:
        """Generate trials for Hyperband algorithm."""
        trials = []
        
        # Hyperband configuration
        max_budget = config.max_budget
        eta = 3  # Reduction factor
        
        # Calculate brackets
        s_max = int(math.log(max_budget / config.min_budget) / math.log(eta))
        
        for s in range(s_max + 1):
            n_configs = int(math.ceil((s_max + 1) / (s + 1)) * eta ** s)
            r = max_budget / (eta ** s)
            
            # Generate random configurations for this bracket
            bracket_trials = []
            for _ in range(min(n_configs, config.max_trials - len(trials))):
                parameters = {}
                
                for param_space in config.parameter_spaces:
                    if param_space.param_type == 'continuous' and param_space.bounds:
                        min_val, max_val = param_space.bounds
                        value = random.uniform(min_val, max_val)
                        
                        if param_space.dtype == 'int':
                            value = int(value)
                        
                        parameters[param_space.name] = value
                    
                    elif param_space.param_type in ['discrete', 'categorical'] and param_space.choices:
                        parameters[param_space.name] = random.choice(param_space.choices)
                
                trial = Trial(
                    id=str(uuid.uuid4()),
                    parameters=parameters,
                    budget=r
                )
                bracket_trials.append(trial)
            
            trials.extend(bracket_trials)
            
            if len(trials) >= config.max_trials:
                break
        
        return trials[:config.max_trials]
    
    async def _run_optimization(
        self,
        optimization_id: str,
        experiment_name: str,
        base_config: TrainingConfig,
        optimization_config: OptimizationConfig,
        trials: List[Trial]
    ):
        """Run the optimization experiment."""
        try:
            logger.info(f"Running optimization {optimization_id} with {len(trials)} trials")
            
            # Keep track of running trials
            running_trials = set()
            completed_trials = []
            
            for trial in trials:
                # Wait if we've hit the concurrent limit
                while len(running_trials) >= optimization_config.max_concurrent_trials:
                    # Check for completed trials
                    completed_trial_ids = []
                    
                    for trial_id in list(running_trials):
                        trial_obj = next((t for t in trials if t.id == trial_id), None)
                        if trial_obj and trial_obj.training_job_id:
                            job_status = await self.training_manager.get_job_status(trial_obj.training_job_id)
                            
                            if job_status and job_status['status'] in ['completed', 'failed', 'cancelled']:
                                completed_trial_ids.append(trial_id)
                                
                                # Extract objective value
                                if job_status['status'] == 'completed' and job_status.get('best_metric'):
                                    trial_obj.objective_value = job_status['best_metric']
                                    trial_obj.status = 'completed'
                                else:
                                    trial_obj.status = 'failed'
                                
                                trial_obj.end_time = datetime.now()
                                completed_trials.append(trial_obj)
                    
                    # Remove completed trials from running set
                    for trial_id in completed_trial_ids:
                        running_trials.discard(trial_id)
                    
                    if len(running_trials) >= optimization_config.max_concurrent_trials:
                        await asyncio.sleep(30)  # Wait 30 seconds before checking again
                
                # Start the trial
                await self._start_trial(trial, experiment_name, base_config)
                running_trials.add(trial.id)
            
            # Wait for remaining trials to complete
            while running_trials:
                completed_trial_ids = []
                
                for trial_id in list(running_trials):
                    trial_obj = next((t for t in trials if t.id == trial_id), None)
                    if trial_obj and trial_obj.training_job_id:
                        job_status = await self.training_manager.get_job_status(trial_obj.training_job_id)
                        
                        if job_status and job_status['status'] in ['completed', 'failed', 'cancelled']:
                            completed_trial_ids.append(trial_id)
                            
                            if job_status['status'] == 'completed' and job_status.get('best_metric'):
                                trial_obj.objective_value = job_status['best_metric']
                                trial_obj.status = 'completed'
                            else:
                                trial_obj.status = 'failed'
                            
                            trial_obj.end_time = datetime.now()
                            completed_trials.append(trial_obj)
                
                for trial_id in completed_trial_ids:
                    running_trials.discard(trial_id)
                
                if running_trials:
                    await asyncio.sleep(30)
            
            # Store optimization results
            self.optimization_history[optimization_id] = trials
            
            # Find best trial
            successful_trials = [t for t in trials if t.status == 'completed' and t.objective_value is not None]
            if successful_trials:
                best_trial = max(successful_trials, key=lambda t: t.objective_value if optimization_config.maximize else -t.objective_value)
                
                logger.info(f"Optimization {optimization_id} completed. Best trial: {best_trial.id}")
                logger.info(f"Best parameters: {best_trial.parameters}")
                logger.info(f"Best objective value: {best_trial.objective_value}")
            else:
                logger.warning(f"Optimization {optimization_id} completed with no successful trials")
            
        except Exception as e:
            logger.error(f"Error in optimization {optimization_id}: {e}")
    
    async def _start_trial(self, trial: Trial, experiment_name: str, base_config: TrainingConfig):
        """Start a single optimization trial."""
        trial.start_time = datetime.now()
        trial.status = 'running'
        
        # Create training config with trial parameters
        trial_config = TrainingConfig(
            model_type=base_config.model_type,
            model_name=f"{base_config.model_name}_trial_{trial.id[:8]}",
            parameters=base_config.parameters.copy(),
            training_data_params=base_config.training_data_params,
            validation_split=base_config.validation_split,
            epochs=int(trial.budget) if base_config.epochs else base_config.epochs,
            batch_size=base_config.batch_size,
            learning_rate=base_config.learning_rate,
            hyperparameters={**base_config.hyperparameters, **trial.parameters}
        )
        
        # Create experiment with single job
        experiment_id = await self.training_manager.create_experiment(
            name=f"{experiment_name}_trial_{trial.id[:8]}",
            description=f"Optimization trial with parameters: {trial.parameters}",
            training_configs=[trial_config]
        )
        
        # Start experiment
        await self.training_manager.start_experiment(experiment_id)
        
        # Get job ID
        experiment = self.training_manager.experiments.get(experiment_id)
        if experiment and experiment.jobs:
            trial.training_job_id = experiment.jobs[0].id
        
        logger.info(f"Started trial {trial.id} with job {trial.training_job_id}")
    
    async def get_optimization_status(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """Get optimization progress and results."""
        if optimization_id not in self.optimization_history:
            return None
        
        trials = self.optimization_history[optimization_id]
        config = self.current_optimizations.get(optimization_id)
        
        total_trials = len(trials)
        completed_trials = [t for t in trials if t.status in ['completed', 'failed']]
        successful_trials = [t for t in trials if t.status == 'completed' and t.objective_value is not None]
        
        progress = len(completed_trials) / total_trials if total_trials > 0 else 0
        
        best_trial = None
        if successful_trials and config:
            best_trial = max(successful_trials, key=lambda t: t.objective_value if config.maximize else -t.objective_value)
        
        return {
            'optimization_id': optimization_id,
            'strategy': config.strategy.value if config else 'unknown',
            'total_trials': total_trials,
            'completed_trials': len(completed_trials),
            'successful_trials': len(successful_trials),
            'failed_trials': len([t for t in trials if t.status == 'failed']),
            'progress': progress,
            'best_trial': {
                'trial_id': best_trial.id,
                'parameters': best_trial.parameters,
                'objective_value': best_trial.objective_value
            } if best_trial else None,
            'is_complete': progress >= 1.0
        }
    
    async def get_optimization_results(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed optimization results."""
        if optimization_id not in self.optimization_history:
            return None
        
        trials = self.optimization_history[optimization_id]
        config = self.current_optimizations.get(optimization_id)
        
        successful_trials = [t for t in trials if t.status == 'completed' and t.objective_value is not None]
        
        if not successful_trials:
            return {
                'optimization_id': optimization_id,
                'trials': [],
                'best_trial': None,
                'convergence_data': []
            }
        
        # Sort trials by objective value
        sorted_trials = sorted(
            successful_trials,
            key=lambda t: t.objective_value if config and config.maximize else -t.objective_value,
            reverse=True
        )
        
        best_trial = sorted_trials[0]
        
        # Create convergence data
        convergence_data = []
        best_so_far = None
        
        for i, trial in enumerate(trials):
            if trial.objective_value is not None:
                if best_so_far is None:
                    best_so_far = trial.objective_value
                else:
                    if config and config.maximize:
                        best_so_far = max(best_so_far, trial.objective_value)
                    else:
                        best_so_far = min(best_so_far, trial.objective_value)
                
                convergence_data.append({
                    'trial_number': i + 1,
                    'best_value': best_so_far,
                    'current_value': trial.objective_value
                })
        
        return {
            'optimization_id': optimization_id,
            'strategy': config.strategy.value if config else 'unknown',
            'best_trial': {
                'trial_id': best_trial.id,
                'parameters': best_trial.parameters,
                'objective_value': best_trial.objective_value,
                'metrics': best_trial.metrics
            },
            'top_trials': [
                {
                    'trial_id': trial.id,
                    'parameters': trial.parameters,
                    'objective_value': trial.objective_value,
                    'rank': i + 1
                }
                for i, trial in enumerate(sorted_trials[:10])
            ],
            'convergence_data': convergence_data,
            'parameter_importance': await self._calculate_parameter_importance(successful_trials),
            'total_trials': len(trials),
            'successful_trials': len(successful_trials)
        }
    
    async def _calculate_parameter_importance(self, trials: List[Trial]) -> Dict[str, float]:
        """Calculate parameter importance based on trial results."""
        if not trials:
            return {}
        
        # Simple correlation-based importance
        parameter_importance = {}
        
        # Get all parameter names
        all_params = set()
        for trial in trials:
            all_params.update(trial.parameters.keys())
        
        for param_name in all_params:
            # Get parameter values and objective values
            param_values = []
            objective_values = []
            
            for trial in trials:
                if param_name in trial.parameters and trial.objective_value is not None:
                    param_val = trial.parameters[param_name]
                    
                    # Convert to numeric if possible
                    if isinstance(param_val, (int, float)):
                        param_values.append(float(param_val))
                        objective_values.append(trial.objective_value)
                    elif isinstance(param_val, str):
                        # For categorical parameters, use hash
                        param_values.append(float(hash(param_val) % 1000))
                        objective_values.append(trial.objective_value)
            
            # Calculate correlation (simplified)
            if len(param_values) > 1 and NUMPY_AVAILABLE:
                correlation = abs(np.corrcoef(param_values, objective_values)[0, 1])
                parameter_importance[param_name] = correlation if not np.isnan(correlation) else 0.0
            else:
                parameter_importance[param_name] = 0.0
        
        return parameter_importance