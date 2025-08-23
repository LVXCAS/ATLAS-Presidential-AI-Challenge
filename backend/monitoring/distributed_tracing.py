"""
Distributed tracing system for tracking requests across microservices.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from contextvars import ContextVar
import json

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.jaeger import JaegerPropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenTelemetry not available - distributed tracing disabled")

from core.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)

# Context variables for trace propagation
trace_context: ContextVar[Optional['TraceContext']] = ContextVar('trace_context', default=None)


@dataclass
class SpanInfo:
    """Information about a trace span."""
    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "success"  # success, error, timeout
    service_name: str = "bloomberg-terminal"


@dataclass
class TraceContext:
    """Trace context for correlation across services."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


class CustomTracer:
    """Custom tracer implementation when OpenTelemetry is not available."""
    
    def __init__(self, service_name: str = "bloomberg-terminal"):
        self.service_name = service_name
        self.active_spans: Dict[str, SpanInfo] = {}
        self.completed_spans: List[SpanInfo] = []
        self.redis = None
        
        logger.info(f"CustomTracer initialized for service: {service_name}")
    
    async def initialize(self):
        """Initialize the custom tracer."""
        self.redis = get_redis_manager().client
        
        # Start background task to persist traces
        asyncio.create_task(self._persistence_loop())
        
        logger.info("CustomTracer initialized")
    
    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Dict[str, Any] = None
    ) -> SpanInfo:
        """Start a new trace span."""
        # Get current trace context
        current_context = trace_context.get()
        
        if current_context and not parent_span_id:
            parent_span_id = current_context.span_id
            trace_id = current_context.trace_id
        else:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        span = SpanInfo(
            span_id=span_id,
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            tags=tags or {},
            service_name=self.service_name
        )
        
        self.active_spans[span_id] = span
        
        # Set trace context
        new_context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
        trace_context.set(new_context)
        
        logger.debug(f"Started span: {operation_name} [{span_id}]")
        
        return span
    
    def finish_span(
        self,
        span: SpanInfo,
        status: str = "success",
        tags: Dict[str, Any] = None
    ):
        """Finish a trace span."""
        if span.span_id not in self.active_spans:
            logger.warning(f"Attempted to finish unknown span: {span.span_id}")
            return
        
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        
        if tags:
            span.tags.update(tags)
        
        # Move from active to completed
        del self.active_spans[span.span_id]
        self.completed_spans.append(span)
        
        logger.debug(f"Finished span: {span.operation_name} [{span.span_id}] - {span.duration_ms:.2f}ms")
    
    def add_span_tag(self, span: SpanInfo, key: str, value: Any):
        """Add a tag to an active span."""
        if span.span_id in self.active_spans:
            span.tags[key] = value
    
    def add_span_log(self, span: SpanInfo, message: str, level: str = "info", fields: Dict[str, Any] = None):
        """Add a log entry to a span."""
        if span.span_id in self.active_spans:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                'fields': fields or {}
            }
            span.logs.append(log_entry)
    
    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        tags: Dict[str, Any] = None,
        parent_span_id: Optional[str] = None
    ):
        """Context manager for tracing operations."""
        span = self.start_span(operation_name, parent_span_id, tags)
        
        try:
            yield span
            self.finish_span(span, "success")
        except Exception as e:
            self.add_span_tag(span, "error", True)
            self.add_span_tag(span, "error.message", str(e))
            self.add_span_log(span, f"Operation failed: {e}", "error")
            self.finish_span(span, "error")
            raise
    
    async def _persistence_loop(self):
        """Persist completed traces to Redis."""
        while True:
            try:
                if self.completed_spans:
                    # Batch persist spans
                    spans_to_persist = self.completed_spans.copy()
                    self.completed_spans.clear()
                    
                    await self._persist_spans(spans_to_persist)
                
                await asyncio.sleep(30)  # Persist every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in trace persistence: {e}")
                await asyncio.sleep(30)
    
    async def _persist_spans(self, spans: List[SpanInfo]):
        """Persist spans to Redis."""
        if not self.redis:
            return
        
        for span in spans:
            span_data = {
                'span_id': span.span_id,
                'parent_span_id': span.parent_span_id,
                'trace_id': span.trace_id,
                'operation_name': span.operation_name,
                'start_time': span.start_time.isoformat(),
                'end_time': span.end_time.isoformat() if span.end_time else None,
                'duration_ms': span.duration_ms,
                'tags': span.tags,
                'logs': span.logs,
                'status': span.status,
                'service_name': span.service_name
            }
            
            # Store span
            await self.redis.setex(
                f"trace:span:{span.span_id}",
                3600,  # 1 hour TTL
                json.dumps(span_data, default=str)
            )
            
            # Add to trace index
            await self.redis.sadd(f"trace:spans:{span.trace_id}", span.span_id)
            await self.redis.expire(f"trace:spans:{span.trace_id}", 3600)
        
        logger.debug(f"Persisted {len(spans)} spans to Redis")
    
    async def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a complete trace by ID."""
        if not self.redis:
            return None
        
        # Get all span IDs for this trace
        span_ids = await self.redis.smembers(f"trace:spans:{trace_id}")
        
        if not span_ids:
            return None
        
        # Retrieve all spans
        spans = []
        for span_id in span_ids:
            span_data = await self.redis.get(f"trace:span:{span_id}")
            if span_data:
                spans.append(json.loads(span_data))
        
        if not spans:
            return None
        
        # Build trace hierarchy
        return self._build_trace_hierarchy(trace_id, spans)
    
    def _build_trace_hierarchy(self, trace_id: str, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build hierarchical trace structure."""
        spans_by_id = {span['span_id']: span for span in spans}
        root_spans = [span for span in spans if not span['parent_span_id']]
        
        def build_span_tree(span):
            children = [
                build_span_tree(child)
                for child in spans
                if child['parent_span_id'] == span['span_id']
            ]
            return {**span, 'children': children}
        
        trace_tree = [build_span_tree(root) for root in root_spans]
        
        # Calculate trace-level metrics
        total_duration = max([span['duration_ms'] or 0 for span in spans])
        error_count = sum(1 for span in spans if span['status'] == 'error')
        
        return {
            'trace_id': trace_id,
            'spans': trace_tree,
            'total_duration_ms': total_duration,
            'span_count': len(spans),
            'error_count': error_count,
            'status': 'error' if error_count > 0 else 'success',
            'services': list(set(span['service_name'] for span in spans))
        }


class DistributedTracing:
    """Main distributed tracing coordinator."""
    
    def __init__(self, service_name: str = "bloomberg-terminal", jaeger_endpoint: str = None):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.tracer = None
        self.custom_tracer = None
        
        logger.info(f"DistributedTracing initialized for service: {service_name}")
    
    async def initialize(self):
        """Initialize distributed tracing."""
        if OPENTELEMETRY_AVAILABLE and self.jaeger_endpoint:
            await self._setup_opentelemetry()
        else:
            await self._setup_custom_tracer()
        
        logger.info("Distributed tracing initialized")
    
    async def _setup_opentelemetry(self):
        """Set up OpenTelemetry with Jaeger."""
        # Create tracer provider
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        # Create Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=14268,
        )
        
        # Create span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Set global propagator
        set_global_textmap(JaegerPropagator())
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        
        logger.info("OpenTelemetry tracing configured with Jaeger")
    
    async def _setup_custom_tracer(self):
        """Set up custom tracing implementation."""
        self.custom_tracer = CustomTracer(self.service_name)
        await self.custom_tracer.initialize()
        
        logger.info("Custom tracing implementation configured")
    
    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        tags: Dict[str, Any] = None,
        parent_span_id: Optional[str] = None
    ):
        """Trace an operation with either OpenTelemetry or custom tracer."""
        if self.tracer and OPENTELEMETRY_AVAILABLE:
            # Use OpenTelemetry
            with self.tracer.start_as_current_span(operation_name) as span:
                if tags:
                    for key, value in tags.items():
                        span.set_attribute(key, value)
                
                try:
                    yield span
                except Exception as e:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        
        elif self.custom_tracer:
            # Use custom tracer
            async with self.custom_tracer.trace_operation(operation_name, tags, parent_span_id) as span:
                yield span
        
        else:
            # No tracing available - yield dummy span
            yield None
    
    async def trace_http_request(
        self,
        method: str,
        url: str,
        status_code: int,
        duration_ms: float,
        user_agent: str = None
    ):
        """Trace an HTTP request."""
        tags = {
            'http.method': method,
            'http.url': url,
            'http.status_code': status_code,
            'http.duration_ms': duration_ms
        }
        
        if user_agent:
            tags['http.user_agent'] = user_agent
        
        async with self.trace_operation(f"HTTP {method}", tags):
            pass
    
    async def trace_database_query(
        self,
        query: str,
        database: str,
        duration_ms: float,
        rows_affected: int = 0
    ):
        """Trace a database query."""
        tags = {
            'db.statement': query[:500],  # Truncate long queries
            'db.name': database,
            'db.duration_ms': duration_ms,
            'db.rows_affected': rows_affected
        }
        
        async with self.trace_operation("DB Query", tags):
            pass
    
    async def trace_external_api_call(
        self,
        service_name: str,
        operation: str,
        duration_ms: float,
        success: bool = True,
        error_message: str = None
    ):
        """Trace an external API call."""
        tags = {
            'external.service': service_name,
            'external.operation': operation,
            'external.duration_ms': duration_ms,
            'external.success': success
        }
        
        if error_message:
            tags['external.error'] = error_message
        
        async with self.trace_operation(f"External API: {service_name}", tags):
            pass
    
    async def trace_agent_execution(
        self,
        agent_name: str,
        operation: str,
        duration_ms: float,
        success: bool = True,
        confidence: float = None
    ):
        """Trace agent execution."""
        tags = {
            'agent.name': agent_name,
            'agent.operation': operation,
            'agent.duration_ms': duration_ms,
            'agent.success': success
        }
        
        if confidence is not None:
            tags['agent.confidence'] = confidence
        
        async with self.trace_operation(f"Agent: {agent_name}", tags):
            pass
    
    async def trace_ml_training(
        self,
        model_type: str,
        model_name: str,
        duration_ms: float,
        accuracy: float = None,
        loss: float = None
    ):
        """Trace ML model training."""
        tags = {
            'ml.model_type': model_type,
            'ml.model_name': model_name,
            'ml.duration_ms': duration_ms
        }
        
        if accuracy is not None:
            tags['ml.accuracy'] = accuracy
        
        if loss is not None:
            tags['ml.loss'] = loss
        
        async with self.trace_operation(f"ML Training: {model_name}", tags):
            pass
    
    async def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a complete trace by ID."""
        if self.custom_tracer:
            return await self.custom_tracer.get_trace(trace_id)
        else:
            logger.warning("Trace retrieval not implemented for OpenTelemetry")
            return None
    
    async def get_recent_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent traces."""
        if not self.custom_tracer or not self.custom_tracer.redis:
            return []
        
        # Get recent trace IDs (this is a simplified implementation)
        # In production, you'd want to maintain a sorted set of traces by timestamp
        trace_keys = await self.custom_tracer.redis.keys("trace:spans:*")
        trace_ids = [key.split(":")[-1] for key in trace_keys[:limit]]
        
        traces = []
        for trace_id in trace_ids:
            trace_data = await self.get_trace(trace_id)
            if trace_data:
                traces.append(trace_data)
        
        return traces
    
    def extract_trace_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Extract trace context from HTTP headers."""
        trace_id = headers.get('x-trace-id')
        span_id = headers.get('x-span-id')
        parent_span_id = headers.get('x-parent-span-id')
        
        if trace_id and span_id:
            return TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id
            )
        
        return None
    
    def inject_trace_context(self, context: TraceContext) -> Dict[str, str]:
        """Inject trace context into HTTP headers."""
        headers = {
            'x-trace-id': context.trace_id,
            'x-span-id': context.span_id
        }
        
        if context.parent_span_id:
            headers['x-parent-span-id'] = context.parent_span_id
        
        return headers