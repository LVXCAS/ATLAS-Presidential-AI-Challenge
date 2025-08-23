"""
ML System API routes for Bloomberg Terminal.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class FeatureResponse(BaseModel):
    """Feature information response."""
    name: str
    type: str
    description: str
    version: str
    tags: List[str]
    dependencies: List[str]
    refresh_interval: int
    created_at: str
    updated_at: str


class FeatureValueResponse(BaseModel):
    """Feature value response."""
    feature_name: str
    symbol: str
    value: Any
    timestamp: str
    status: str
    computation_time_ms: float
    version: str


class FeatureStatsResponse(BaseModel):
    """Feature statistics response."""
    feature_name: str
    feature_type: str
    description: str
    cached_symbols: int
    fresh_values: int
    stale_values: int
    quality_stats: Dict[str, Any]
    last_updated: str


class ModelResponse(BaseModel):
    """ML model information response."""
    name: str
    type: str
    prediction_type: str
    algorithm: str
    status: str
    version: str
    feature_count: int
    created_at: str
    performance: Optional[Dict[str, Any]]


class PredictionResponse(BaseModel):
    """Model prediction response."""
    id: str
    model_name: str
    symbol: str
    prediction_type: str
    value: Any
    confidence: float
    timestamp: str
    metadata: Dict[str, Any]


class ModelMetricsResponse(BaseModel):
    """Model performance metrics response."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    validation_samples: int
    feature_importance: Dict[str, float]
    timestamp: str


class PredictionRequest(BaseModel):
    """Prediction request model."""
    model_name: str
    symbol: str
    features: Optional[Dict[str, Any]] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    model_name: str
    symbols: List[str]


class ModelTrainingRequest(BaseModel):
    """Model training request."""
    model_name: str
    symbols: Optional[List[str]] = None
    force_retrain: bool = False


# Feature Store Endpoints

@router.get("/features", response_model=List[FeatureResponse])
async def list_features(tags: Optional[str] = Query(None, description="Comma-separated tags to filter by")):
    """List all available features with optional tag filtering."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.feature_store:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        tag_list = tags.split(',') if tags else None
        features = await ml_service.feature_store.list_features(tag_list)
        
        return [FeatureResponse(**feature) for feature in features]
        
    except Exception as e:
        logger.error(f"Error listing features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{feature_name}/stats", response_model=FeatureStatsResponse)
async def get_feature_stats(feature_name: str = Path(..., description="Name of the feature")):
    """Get statistics and quality metrics for a specific feature."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.feature_store:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        stats = await ml_service.feature_store.get_feature_stats(feature_name)
        
        if not stats:
            raise HTTPException(status_code=404, detail=f"Feature {feature_name} not found")
        
        return FeatureStatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{feature_name}/{symbol}", response_model=FeatureValueResponse)
async def get_feature_value(
    feature_name: str = Path(..., description="Name of the feature"),
    symbol: str = Path(..., description="Trading symbol"),
    force_refresh: bool = Query(False, description="Force recomputation")
):
    """Get current value of a feature for a specific symbol."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.feature_store:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        feature_value = await ml_service.feature_store.get_feature(feature_name, symbol, force_refresh)
        
        if not feature_value:
            raise HTTPException(status_code=404, detail=f"Feature {feature_name} not available for {symbol}")
        
        return FeatureValueResponse(
            feature_name=feature_value.feature_name,
            symbol=feature_value.symbol,
            value=feature_value.value,
            timestamp=feature_value.timestamp.isoformat(),
            status=feature_value.status.value,
            computation_time_ms=feature_value.computation_time_ms,
            version=feature_value.version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature value: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/vector/{symbol}")
async def get_feature_vector(
    symbol: str = Path(..., description="Trading symbol"),
    features: Optional[str] = Query(None, description="Comma-separated feature names"),
    include_metadata: bool = Query(False, description="Include feature metadata")
) -> Dict[str, Any]:
    """Get feature vector for a symbol (used for ML model input)."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.feature_store:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        feature_names = features.split(',') if features else None
        
        feature_vector = await ml_service.feature_store.get_feature_vector(
            symbol, feature_names, include_metadata
        )
        
        return feature_vector
        
    except Exception as e:
        logger.error(f"Error getting feature vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/historical")
async def get_historical_features(
    features: str = Query(..., description="Comma-separated feature names"),
    symbols: str = Query(..., description="Comma-separated trading symbols"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    interval: str = Query("1h", description="Data interval (1m, 5m, 1h, 1d)")
) -> Dict[str, Any]:
    """Get historical feature data for backtesting and analysis."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.feature_store:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        feature_names = features.split(',')
        symbol_list = symbols.split(',')
        
        # Parse dates
        start_time = datetime.fromisoformat(start_date)
        end_time = datetime.fromisoformat(end_date)
        
        # Validate date range
        if (end_time - start_time).days > 365:
            raise HTTPException(status_code=400, detail="Date range cannot exceed 365 days")
        
        historical_data = await ml_service.feature_store.get_historical_features(
            feature_names, symbol_list, start_time, end_time, interval
        )
        
        # Convert DataFrame to dict for JSON response
        return {
            'data': historical_data.to_dict('records'),
            'shape': list(historical_data.shape),
            'columns': list(historical_data.columns),
            'start_date': start_date,
            'end_date': end_date,
            'interval': interval
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Error getting historical features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/{feature_name}/invalidate")
async def invalidate_feature(
    feature_name: str = Path(..., description="Name of the feature"),
    symbol: Optional[str] = Query(None, description="Specific symbol to invalidate")
) -> Dict[str, str]:
    """Invalidate cached feature values to force recomputation."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.feature_store:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        success = await ml_service.feature_store.invalidate_feature(feature_name, symbol)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to invalidate feature {feature_name}")
        
        message = f"Invalidated feature {feature_name}"
        if symbol:
            message += f" for symbol {symbol}"
        
        return {
            'status': 'success',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invalidating feature: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/metrics")
async def get_feature_store_metrics() -> Dict[str, Any]:
    """Get feature store performance metrics."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.feature_store:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        metrics = await ml_service.feature_store.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting feature store metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ML Pipeline Endpoints

@router.get("/models", response_model=List[ModelResponse])
async def list_models():
    """List all registered ML models."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.ml_pipeline:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        models = await ml_service.ml_pipeline.list_models()
        return [ModelResponse(**model) for model in models]
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(model_name: str = Path(..., description="Name of the model")):
    """Get performance metrics for a specific model."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.ml_pipeline:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        metrics = await ml_service.ml_pipeline.get_model_performance(model_name)
        
        if not metrics:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found or not trained")
        
        return ModelMetricsResponse(
            model_name=metrics.model_name,
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1_score=metrics.f1_score,
            training_samples=metrics.training_samples,
            validation_samples=metrics.validation_samples,
            feature_importance=metrics.feature_importance,
            timestamp=metrics.timestamp.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/status")
async def get_model_status(model_name: str = Path(..., description="Name of the model")) -> Dict[str, str]:
    """Get current status of a model."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.ml_pipeline:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        status = await ml_service.ml_pipeline.get_model_status(model_name)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        return {
            'model_name': model_name,
            'status': status.value,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/train")
async def train_model(
    model_name: str = Path(..., description="Name of the model"),
    request: ModelTrainingRequest
) -> Dict[str, Any]:
    """Train or retrain a model."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.ml_pipeline:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        success = await ml_service.ml_pipeline.train_model(request.model_name, request.symbols)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Training failed for model {model_name}")
        
        return {
            'status': 'success',
            'message': f'Training started for model {model_name}',
            'model_name': model_name,
            'symbols': request.symbols or [],
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/deploy")
async def deploy_model(model_name: str = Path(..., description="Name of the model")) -> Dict[str, str]:
    """Deploy a trained model for inference."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.ml_pipeline:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        success = await ml_service.ml_pipeline.deploy_model(model_name)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Deployment failed for model {model_name}")
        
        return {
            'status': 'success',
            'message': f'Model {model_name} deployed successfully',
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate a prediction using a deployed model."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.ml_pipeline:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        prediction = await ml_service.ml_pipeline.predict(
            request.model_name,
            request.symbol,
            request.features
        )
        
        if not prediction:
            raise HTTPException(status_code=400, detail=f"Prediction failed for {request.model_name}")
        
        return PredictionResponse(
            id=prediction.id,
            model_name=prediction.model_name,
            symbol=prediction.symbol,
            prediction_type=prediction.prediction_type.value,
            value=prediction.value,
            confidence=prediction.confidence,
            timestamp=prediction.timestamp.isoformat(),
            metadata=prediction.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest) -> Dict[str, Any]:
    """Generate predictions for multiple symbols efficiently."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service or not ml_service.ml_pipeline:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        predictions = await ml_service.ml_pipeline.batch_predict(
            request.model_name,
            request.symbols
        )
        
        # Format response
        results = {}
        for symbol, prediction in predictions.items():
            if prediction:
                results[symbol] = {
                    'id': prediction.id,
                    'value': prediction.value,
                    'confidence': prediction.confidence,
                    'timestamp': prediction.timestamp.isoformat()
                }
            else:
                results[symbol] = None
        
        return {
            'model_name': request.model_name,
            'predictions': results,
            'total_symbols': len(request.symbols),
            'successful_predictions': len([p for p in results.values() if p is not None]),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/history/{symbol}")
async def get_prediction_history(
    symbol: str = Path(..., description="Trading symbol"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    limit: int = Query(100, description="Maximum number of predictions to return")
) -> Dict[str, Any]:
    """Get prediction history for a symbol."""
    try:
        # This would query a database of historical predictions
        # For now, return mock data
        
        return {
            'symbol': symbol,
            'model_name': model_name,
            'predictions': [],  # Would contain historical predictions
            'count': 0,
            'message': 'Prediction history not yet implemented - would query historical database'
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/metrics")
async def get_pipeline_metrics() -> Dict[str, Any]:
    """Get ML pipeline performance metrics."""
    try:
        from services.ml_service import ml_service
        
        if not ml_service:
            raise HTTPException(status_code=503, detail="ML service not available")
        
        metrics = {}
        
        if ml_service.ml_pipeline:
            pipeline_metrics = await ml_service.ml_pipeline.get_pipeline_metrics()
            metrics['pipeline'] = pipeline_metrics
        
        if ml_service.feature_store:
            feature_metrics = await ml_service.feature_store.get_metrics()
            metrics['feature_store'] = feature_metrics
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def ml_health_check() -> Dict[str, Any]:
    """Health check for ML system."""
    try:
        from services.ml_service import ml_service
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        if not ml_service:
            health_data['status'] = 'unhealthy'
            health_data['error'] = 'ML service not available'
            return health_data
        
        # Check feature store
        if ml_service.feature_store and ml_service.feature_store.is_running:
            health_data['components']['feature_store'] = 'healthy'
            
            # Get feature store metrics
            fs_metrics = await ml_service.feature_store.get_metrics()
            health_data['components']['feature_store_metrics'] = {
                'registered_features': fs_metrics.get('registered_features', 0),
                'cached_features': fs_metrics.get('cached_features', 0),
                'cache_hit_rate': fs_metrics.get('cache_hit_rate', 0)
            }
        else:
            health_data['components']['feature_store'] = 'unhealthy'
            health_data['status'] = 'degraded'
        
        # Check ML pipeline
        if ml_service.ml_pipeline and ml_service.ml_pipeline.is_running:
            health_data['components']['ml_pipeline'] = 'healthy'
            
            # Get pipeline metrics
            pipeline_metrics = await ml_service.ml_pipeline.get_pipeline_metrics()
            health_data['components']['ml_pipeline_metrics'] = {
                'total_models': pipeline_metrics.get('total_models', 0),
                'deployed_models': pipeline_metrics.get('deployed_models', 0),
                'average_accuracy': pipeline_metrics.get('average_accuracy', 0)
            }
        else:
            health_data['components']['ml_pipeline'] = 'unhealthy'
            health_data['status'] = 'degraded'
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error in ML health check: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }