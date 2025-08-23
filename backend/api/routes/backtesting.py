"""
Backtesting API routes for Bloomberg Terminal.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import uuid

from backtesting import (
    BacktestEngine, BacktestResults, DataLoader, StrategyTester, 
    ParameterRange, create_data_feed
)
from agents.base_agent import TradingSignal, SignalType

logger = logging.getLogger(__name__)

router = APIRouter()

# Global storage for backtest results and running jobs
backtest_jobs: Dict[str, Dict[str, Any]] = {}
backtest_results_cache: Dict[str, BacktestResults] = {}


class BacktestRequest(BaseModel):
    """Backtest request model."""
    name: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = Field(default=1000000.0, ge=10000)
    strategy_type: str = Field(default="buy_and_hold")
    strategy_parameters: Dict[str, Any] = Field(default_factory=dict)
    frequency: str = Field(default="1d")
    benchmark: Optional[str] = None


class OptimizationRequest(BaseModel):
    """Parameter optimization request model."""
    name: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = Field(default=1000000.0, ge=10000)
    strategy_type: str
    parameter_ranges: List[Dict[str, Any]]
    optimization_metric: str = Field(default="sharpe_ratio")
    max_combinations: int = Field(default=1000, ge=1, le=10000)


class WalkForwardRequest(BaseModel):
    """Walk-forward analysis request model."""
    name: str
    symbols: List[str] 
    start_date: str
    end_date: str
    initial_capital: float = Field(default=1000000.0, ge=10000)
    strategy_type: str
    parameter_ranges: List[Dict[str, Any]]
    in_sample_months: int = Field(default=12, ge=1, le=60)
    out_sample_months: int = Field(default=3, ge=1, le=12)
    step_months: int = Field(default=1, ge=1, le=12)


class BacktestJobResponse(BaseModel):
    """Backtest job response model."""
    job_id: str
    status: str
    created_at: str
    progress: float
    estimated_completion: Optional[str]


class BacktestResultsResponse(BaseModel):
    """Backtest results response model."""
    job_id: str
    name: str
    status: str
    results: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]
    equity_curve: Optional[List[Dict[str, Any]]]
    trades: Optional[List[Dict[str, Any]]]
    error: Optional[str]


def create_simple_strategy(strategy_type: str, parameters: Dict[str, Any]):
    """Create a simple strategy function for backtesting."""
    
    def buy_and_hold_strategy(timestamp: datetime, context: Dict[str, Any]) -> List[TradingSignal]:
        """Simple buy and hold strategy."""
        signals = []
        portfolio = context['portfolio']
        
        # Only buy at the beginning if we have no positions
        if not portfolio.positions:
            for symbol in context.get('symbols', []):
                if symbol in context['current_prices']:
                    signal = TradingSignal(
                        id=str(uuid.uuid4()),
                        agent_name="BuyAndHold",
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.BUY,
                        confidence=1.0,
                        strength=1.0,
                        reasoning={"strategy": "buy_and_hold"},
                        features_used={},
                        prediction_horizon=999999,  # Very long term
                        target_price=context['current_prices'][symbol] * 1.1,
                        stop_loss=context['current_prices'][symbol] * 0.9,
                        risk_score=0.2,
                        expected_return=0.08
                    )
                    signals.append(signal)
        
        return signals
    
    def mean_reversion_strategy(timestamp: datetime, context: Dict[str, Any]) -> List[TradingSignal]:
        """Simple mean reversion strategy."""
        signals = []
        parameters = context.get('parameters', {})
        
        # Strategy parameters
        lookback_period = parameters.get('lookback_period', 20)
        entry_threshold = parameters.get('entry_threshold', 2.0)
        exit_threshold = parameters.get('exit_threshold', 0.5)
        
        # This is a simplified version - would need historical data for real implementation
        for symbol in context.get('symbols', []):
            if symbol in context['current_prices']:
                current_price = context['current_prices'][symbol]
                
                # Mock mean reversion logic (would need actual price history)
                # Generate random signals for demonstration
                import random
                random.seed(int(timestamp.timestamp()) + hash(symbol))
                
                if random.random() < 0.1:  # 10% chance of signal
                    signal_type = SignalType.BUY if random.random() < 0.5 else SignalType.SELL
                    
                    signal = TradingSignal(
                        id=str(uuid.uuid4()),
                        agent_name="MeanReversion",
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=signal_type,
                        confidence=0.7,
                        strength=0.8,
                        reasoning={"strategy": "mean_reversion", "threshold": entry_threshold},
                        features_used={"lookback": lookback_period},
                        prediction_horizon=5,
                        target_price=current_price * (1.05 if signal_type == SignalType.BUY else 0.95),
                        stop_loss=current_price * (0.98 if signal_type == SignalType.BUY else 1.02),
                        risk_score=0.4,
                        expected_return=0.05
                    )
                    signals.append(signal)
        
        return signals
    
    def momentum_strategy(timestamp: datetime, context: Dict[str, Any]) -> List[TradingSignal]:
        """Simple momentum strategy."""
        signals = []
        parameters = context.get('parameters', {})
        
        # Strategy parameters
        momentum_period = parameters.get('momentum_period', 10)
        momentum_threshold = parameters.get('momentum_threshold', 0.05)
        
        # Mock momentum logic
        for symbol in context.get('symbols', []):
            if symbol in context['current_prices']:
                current_price = context['current_prices'][symbol]
                
                # Generate momentum-based signals (simplified)
                import random
                random.seed(int(timestamp.timestamp()) + hash(symbol) + 123)
                
                momentum = random.uniform(-0.1, 0.1)  # Mock momentum
                
                if abs(momentum) > momentum_threshold:
                    signal_type = SignalType.BUY if momentum > 0 else SignalType.SELL
                    
                    signal = TradingSignal(
                        id=str(uuid.uuid4()),
                        agent_name="Momentum",
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=signal_type,
                        confidence=min(abs(momentum) * 10, 1.0),
                        strength=0.6,
                        reasoning={"strategy": "momentum", "momentum": momentum},
                        features_used={"momentum_period": momentum_period},
                        prediction_horizon=3,
                        target_price=current_price * (1.03 if signal_type == SignalType.BUY else 0.97),
                        stop_loss=current_price * (0.97 if signal_type == SignalType.BUY else 1.03),
                        risk_score=0.5,
                        expected_return=0.03
                    )
                    signals.append(signal)
        
        return signals
    
    # Return appropriate strategy function
    if strategy_type == "buy_and_hold":
        return buy_and_hold_strategy
    elif strategy_type == "mean_reversion":
        return mean_reversion_strategy
    elif strategy_type == "momentum":
        return momentum_strategy
    else:
        return buy_and_hold_strategy  # Default


@router.post("/backtest", response_model=BacktestJobResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start a new backtest."""
    try:
        job_id = str(uuid.uuid4())
        
        # Validate dates
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Create job record
        backtest_jobs[job_id] = {
            'id': job_id,
            'name': request.name,
            'status': 'queued',
            'progress': 0.0,
            'created_at': datetime.now().isoformat(),
            'request': request.dict(),
            'error': None
        }
        
        # Start backtest in background
        background_tasks.add_task(
            _run_backtest_job,
            job_id,
            request
        )
        
        return BacktestJobResponse(
            job_id=job_id,
            status='queued',
            created_at=backtest_jobs[job_id]['created_at'],
            progress=0.0,
            estimated_completion=None
        )
        
    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", response_model=BacktestJobResponse)
async def run_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start parameter optimization."""
    try:
        job_id = str(uuid.uuid4())
        
        # Validate dates
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Create job record
        backtest_jobs[job_id] = {
            'id': job_id,
            'name': request.name,
            'status': 'queued',
            'progress': 0.0,
            'created_at': datetime.now().isoformat(),
            'request': request.dict(),
            'type': 'optimization',
            'error': None
        }
        
        # Start optimization in background
        background_tasks.add_task(
            _run_optimization_job,
            job_id,
            request
        )
        
        return BacktestJobResponse(
            job_id=job_id,
            status='queued',
            created_at=backtest_jobs[job_id]['created_at'],
            progress=0.0,
            estimated_completion=None
        )
        
    except Exception as e:
        logger.error(f"Error starting optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/walk-forward", response_model=BacktestJobResponse)
async def run_walk_forward(request: WalkForwardRequest, background_tasks: BackgroundTasks):
    """Start walk-forward analysis."""
    try:
        job_id = str(uuid.uuid4())
        
        # Validate dates
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Create job record
        backtest_jobs[job_id] = {
            'id': job_id,
            'name': request.name,
            'status': 'queued',
            'progress': 0.0,
            'created_at': datetime.now().isoformat(),
            'request': request.dict(),
            'type': 'walk_forward',
            'error': None
        }
        
        # Start walk-forward analysis in background
        background_tasks.add_task(
            _run_walk_forward_job,
            job_id,
            request
        )
        
        return BacktestJobResponse(
            job_id=job_id,
            status='queued',
            created_at=backtest_jobs[job_id]['created_at'],
            progress=0.0,
            estimated_completion=None
        )
        
    except Exception as e:
        logger.error(f"Error starting walk-forward analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=List[BacktestJobResponse])
async def get_backtest_jobs():
    """Get all backtest jobs."""
    try:
        jobs = []
        for job_data in backtest_jobs.values():
            jobs.append(BacktestJobResponse(
                job_id=job_data['id'],
                status=job_data['status'],
                created_at=job_data['created_at'],
                progress=job_data['progress'],
                estimated_completion=job_data.get('estimated_completion')
            ))
        
        return jobs
        
    except Exception as e:
        logger.error(f"Error getting backtest jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=BacktestJobResponse)
async def get_backtest_job(job_id: str):
    """Get specific backtest job status."""
    try:
        if job_id not in backtest_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_data = backtest_jobs[job_id]
        
        return BacktestJobResponse(
            job_id=job_data['id'],
            status=job_data['status'],
            created_at=job_data['created_at'],
            progress=job_data['progress'],
            estimated_completion=job_data.get('estimated_completion')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{job_id}", response_model=BacktestResultsResponse)
async def get_backtest_results(job_id: str):
    """Get backtest results."""
    try:
        if job_id not in backtest_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_data = backtest_jobs[job_id]
        
        response_data = {
            'job_id': job_id,
            'name': job_data['name'],
            'status': job_data['status'],
            'results': None,
            'performance_metrics': None,
            'equity_curve': None,
            'trades': None,
            'error': job_data.get('error')
        }
        
        if job_data['status'] == 'completed' and job_id in backtest_results_cache:
            results = backtest_results_cache[job_id]
            
            if hasattr(results, 'dict'):
                # Handle results object with dict method
                response_data['results'] = results.dict()
            else:
                # Handle results as dictionary or other format
                response_data['results'] = results
                
            if hasattr(results, 'performance_metrics'):
                response_data['performance_metrics'] = results.performance_metrics
                
            if hasattr(results, 'equity_curve'):
                response_data['equity_curve'] = [
                    {'timestamp': timestamp.isoformat(), 'equity': equity}
                    for timestamp, equity in results.equity_curve
                ]
                
            if hasattr(results, 'trades'):
                response_data['trades'] = [
                    {
                        'id': trade.id,
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'timestamp': trade.timestamp.isoformat(),
                        'commission': trade.commission,
                        'pnl': trade.pnl
                    }
                    for trade in results.trades[:100]  # Limit to first 100 trades
                ]
        
        return BacktestResultsResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest results {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def cancel_backtest_job(job_id: str):
    """Cancel a running backtest job."""
    try:
        if job_id not in backtest_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_data = backtest_jobs[job_id]
        
        if job_data['status'] in ['completed', 'failed', 'cancelled']:
            raise HTTPException(status_code=400, detail=f"Job {job_id} is already {job_data['status']}")
        
        # Mark job as cancelled
        backtest_jobs[job_id]['status'] = 'cancelled'
        
        return {'message': f'Job {job_id} cancelled successfully'}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def get_available_strategies():
    """Get list of available trading strategies."""
    strategies = [
        {
            'name': 'buy_and_hold',
            'display_name': 'Buy and Hold',
            'description': 'Simple buy and hold strategy for long-term investing',
            'parameters': []
        },
        {
            'name': 'mean_reversion',
            'display_name': 'Mean Reversion',
            'description': 'Mean reversion strategy based on price deviation from moving average',
            'parameters': [
                {'name': 'lookback_period', 'type': 'int', 'min': 5, 'max': 100, 'default': 20},
                {'name': 'entry_threshold', 'type': 'float', 'min': 0.5, 'max': 5.0, 'default': 2.0},
                {'name': 'exit_threshold', 'type': 'float', 'min': 0.1, 'max': 2.0, 'default': 0.5}
            ]
        },
        {
            'name': 'momentum',
            'display_name': 'Momentum',
            'description': 'Momentum strategy that follows price trends',
            'parameters': [
                {'name': 'momentum_period', 'type': 'int', 'min': 3, 'max': 50, 'default': 10},
                {'name': 'momentum_threshold', 'type': 'float', 'min': 0.01, 'max': 0.2, 'default': 0.05}
            ]
        }
    ]
    
    return strategies


async def _run_backtest_job(job_id: str, request: BacktestRequest):
    """Run backtest job in background."""
    try:
        # Update job status
        backtest_jobs[job_id]['status'] = 'running'
        backtest_jobs[job_id]['progress'] = 0.1
        
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Create strategy function
        strategy_function = create_simple_strategy(request.strategy_type, request.strategy_parameters)
        
        # Update progress
        backtest_jobs[job_id]['progress'] = 0.2
        
        # Create and configure backtest engine
        engine = BacktestEngine(request.initial_capital)
        
        # Create data feed
        data_feed = await create_data_feed(
            request.symbols,
            start_date,
            end_date,
            request.frequency
        )
        engine.add_data_feed(data_feed)
        
        # Update progress
        backtest_jobs[job_id]['progress'] = 0.4
        
        # Add symbols to strategy context
        def contextualized_strategy(timestamp: datetime, context: Dict[str, Any]) -> List[TradingSignal]:
            context['symbols'] = request.symbols
            return strategy_function(timestamp, context)
        
        # Run backtest
        backtest_jobs[job_id]['progress'] = 0.6
        results = await engine.run_backtest(
            start_date,
            end_date,
            contextualized_strategy,
            request.symbols,
            request.frequency
        )
        
        # Store results
        backtest_results_cache[job_id] = results
        
        # Update job status
        backtest_jobs[job_id]['status'] = 'completed'
        backtest_jobs[job_id]['progress'] = 1.0
        backtest_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
        logger.info(f"Backtest job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in backtest job {job_id}: {e}")
        backtest_jobs[job_id]['status'] = 'failed'
        backtest_jobs[job_id]['error'] = str(e)


async def _run_optimization_job(job_id: str, request: OptimizationRequest):
    """Run optimization job in background."""
    try:
        # Update job status
        backtest_jobs[job_id]['status'] = 'running'
        backtest_jobs[job_id]['progress'] = 0.1
        
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Create parameter ranges
        param_ranges = []
        for param_config in request.parameter_ranges:
            param_range = ParameterRange(
                name=param_config['name'],
                min_value=param_config['min_value'],
                max_value=param_config['max_value'],
                step=param_config['step'],
                param_type=param_config.get('type', 'float')
            )
            param_ranges.append(param_range)
        
        # Update progress
        backtest_jobs[job_id]['progress'] = 0.2
        
        # Create strategy function
        def strategy_function(timestamp: datetime, context: Dict[str, Any]) -> List[TradingSignal]:
            context['symbols'] = request.symbols
            base_strategy = create_simple_strategy(request.strategy_type, context.get('parameters', {}))
            return base_strategy(timestamp, context)
        
        # Create strategy tester
        config = {'optimization_metric': request.optimization_metric}
        tester = StrategyTester(config)
        
        # Update progress
        backtest_jobs[job_id]['progress'] = 0.4
        
        # Run optimization
        optimization_results = await tester.optimize_parameters(
            strategy_function,
            param_ranges,
            request.symbols,
            start_date,
            end_date,
            request.initial_capital,
            request.max_combinations
        )
        
        # Store results
        backtest_results_cache[job_id] = {
            'type': 'optimization',
            'results': optimization_results[:50],  # Limit to top 50
            'best_parameters': optimization_results[0].parameters if optimization_results else {},
            'total_combinations': len(optimization_results)
        }
        
        # Update job status
        backtest_jobs[job_id]['status'] = 'completed'
        backtest_jobs[job_id]['progress'] = 1.0
        backtest_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
        logger.info(f"Optimization job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in optimization job {job_id}: {e}")
        backtest_jobs[job_id]['status'] = 'failed'
        backtest_jobs[job_id]['error'] = str(e)


async def _run_walk_forward_job(job_id: str, request: WalkForwardRequest):
    """Run walk-forward analysis job in background."""
    try:
        # Update job status
        backtest_jobs[job_id]['status'] = 'running'
        backtest_jobs[job_id]['progress'] = 0.1
        
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Create parameter ranges
        param_ranges = []
        for param_config in request.parameter_ranges:
            param_range = ParameterRange(
                name=param_config['name'],
                min_value=param_config['min_value'],
                max_value=param_config['max_value'],
                step=param_config['step'],
                param_type=param_config.get('type', 'float')
            )
            param_ranges.append(param_range)
        
        # Update progress
        backtest_jobs[job_id]['progress'] = 0.2
        
        # Create strategy function
        def strategy_function(timestamp: datetime, context: Dict[str, Any]) -> List[TradingSignal]:
            context['symbols'] = request.symbols
            base_strategy = create_simple_strategy(request.strategy_type, context.get('parameters', {}))
            return base_strategy(timestamp, context)
        
        # Create strategy tester
        tester = StrategyTester()
        
        # Update progress
        backtest_jobs[job_id]['progress'] = 0.4
        
        # Run walk-forward analysis
        wf_results = await tester.walk_forward_analysis(
            strategy_function,
            param_ranges,
            request.symbols,
            start_date,
            end_date,
            request.in_sample_months,
            request.out_sample_months,
            request.step_months,
            request.initial_capital
        )
        
        # Store results
        backtest_results_cache[job_id] = {
            'type': 'walk_forward',
            'results': wf_results,
            'total_periods': len(wf_results)
        }
        
        # Update job status
        backtest_jobs[job_id]['status'] = 'completed'
        backtest_jobs[job_id]['progress'] = 1.0
        backtest_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
        logger.info(f"Walk-forward job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in walk-forward job {job_id}: {e}")
        backtest_jobs[job_id]['status'] = 'failed'
        backtest_jobs[job_id]['error'] = str(e)