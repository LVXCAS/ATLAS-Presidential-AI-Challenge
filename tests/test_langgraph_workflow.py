"""
Test suite for LangGraph Workflow Implementation

Tests the core workflow functionality, agent coordination, communication protocols,
and monitoring systems.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import json

from agents.langgraph_workflow import (
    LangGraphTradingWorkflow,
    TradingSystemState,
    MarketRegime,
    WorkflowPhase,
    MarketData,
    Signal,
    Order,
    RiskMetrics,
    Alert,
    ConditionalRouter,
    WorkflowOrchestrator,
    create_trading_workflow,
    create_workflow_orchestrator
)
from agents.communication_protocols import (
    MessageBus,
    AgentCoordinator,
    ConflictResolver,
    Message,
    MessageType,
    MessagePriority,
    AgentRole,
    create_message_bus,
    create_agent_coordinator,
    create_conflict_resolver
)
from agents.workflow_monitoring import (
    WorkflowMonitor,
    MonitoringLevel,
    MetricsCollector,
    EventLogger,
    AlertManager,
    WorkflowDebugger,
    MonitoredExecution,
    create_workflow_monitor
)


class TestLangGraphWorkflow:
    """Test cases for LangGraph workflow implementation"""
    
    @pytest.fixture
    def workflow(self):
        """Create a workflow instance for testing"""
        with patch('agents.langgraph_workflow.MarketDataIngestor'), \
             patch('agents.langgraph_workflow.NewsSentimentAgent'), \
             patch('agents.langgraph_workflow.MomentumTradingAgent'), \
             patch('agents.langgraph_workflow.MeanReversionAgent'), \
             patch('agents.langgraph_workflow.OptionsVolatilityAgent'), \
             patch('agents.langgraph_workflow.PortfolioAllocatorAgent'), \
             patch('agents.langgraph_workflow.RiskManagerAgent'), \
             patch('agents.langgraph_workflow.ExecutionEngineAgent'):
            return LangGraphTradingWorkflow()
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample trading system state"""
        return TradingSystemState(
            market_data={
                "AAPL": MarketData(
                    symbol="AAPL",
                    timestamp=datetime.now(),
                    open=150.0,
                    high=152.0,
                    low=149.0,
                    close=151.0,
                    volume=1000000
                )
            },
            historical_data={},
            news_articles=[],
            sentiment_scores={"AAPL": 0.7},
            market_events=[],
            raw_signals={},
            fused_signals={},
            signal_conflicts=[],
            portfolio_state={},
            positions={},
            risk_metrics=RiskMetrics(
                portfolio_value=100000.0,
                daily_pnl=1000.0,
                var_95=5000.0,
                max_drawdown=2000.0,
                position_count=5,
                leverage=1.2,
                risk_score=0.3
            ),
            risk_limits={"max_position_size": 10000},
            pending_orders=[],
            executed_orders=[],
            execution_reports=[],
            market_regime=MarketRegime.NORMAL,
            workflow_phase=WorkflowPhase.DATA_INGESTION,
            system_alerts=[],
            performance_metrics={},
            symbols_universe=["AAPL", "GOOGL", "MSFT"],
            active_strategies=["momentum", "mean_reversion"],
            model_versions={"momentum": "1.0.0"},
            agent_states={},
            execution_log=[],
            error_log=[]
        )
    
    def test_workflow_initialization(self, workflow):
        """Test workflow initialization"""
        assert workflow is not None
        assert workflow.graph is not None
        assert len(workflow.agents) > 0
        assert workflow.checkpointer is not None
        
        status = workflow.get_workflow_status()
        assert status["graph_compiled"] is True
        assert status["agents_initialized"] > 0
    
    def test_market_regime_detection(self, workflow):
        """Test market regime detection"""
        # Test high volatility detection
        high_vol_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=100.0,
                high=110.0,  # 10% range
                low=95.0,
                close=105.0,
                volume=1000000
            )
        }
        
        regime = workflow._detect_market_regime(high_vol_data)
        assert regime == MarketRegime.HIGH_VOLATILITY
        
        # Test low volatility detection
        low_vol_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=100.0,
                high=100.5,  # 0.5% range
                low=99.8,
                close=100.2,
                volume=1000000
            )
        }
        
        regime = workflow._detect_market_regime(low_vol_data)
        assert regime == MarketRegime.LOW_VOLATILITY
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, workflow, sample_state):
        """Test complete workflow execution"""
        # Mock agent methods
        for agent_name, agent in workflow.agents.items():
            if hasattr(agent, 'get_latest_data'):
                agent.get_latest_data = AsyncMock(return_value={
                    "open": 150.0, "high": 152.0, "low": 149.0, 
                    "close": 151.0, "volume": 1000000
                })
            if hasattr(agent, 'analyze_sentiment'):
                agent.analyze_sentiment = AsyncMock(return_value={"sentiment_score": 0.7})
            if hasattr(agent, 'generate_signal'):
                agent.generate_signal = AsyncMock(return_value={
                    "signal_strength": 0.8,
                    "confidence": 0.9,
                    "top_3_reasons": ["Strong momentum", "High volume", "Positive sentiment"]
                })
            if hasattr(agent, 'fuse_signals'):
                agent.fuse_signals = AsyncMock(return_value={
                    "signal_strength": 0.75,
                    "confidence": 0.85,
                    "top_3_reasons": ["Multi-strategy consensus", "High confidence", "Good risk-reward"]
                })
            if hasattr(agent, 'assess_portfolio_risk'):
                agent.assess_portfolio_risk = AsyncMock(return_value={
                    "portfolio_value": 100000.0,
                    "risk_score": 0.3,
                    "var_95": 5000.0
                })
            if hasattr(agent, 'create_order'):
                agent.create_order = AsyncMock(return_value={
                    "quantity": 100,
                    "order_type": "MARKET"
                })
        
        # Run workflow
        result = await workflow.run_workflow(sample_state)
        
        # Verify results
        assert result is not None
        assert "market_data" in result
        assert "workflow_phase" in result
        assert len(result.get("execution_log", [])) > 0
    
    def test_conditional_routing(self, workflow, sample_state):
        """Test conditional routing logic"""
        # Test routing to portfolio allocator
        sample_state["agent_states"] = {
            "momentum_agent": {"status": "completed"},
            "mean_reversion_agent": {"status": "completed"},
            "options_agent": {"status": "completed"}
        }
        
        route = workflow._route_to_portfolio_or_continue(sample_state)
        assert route == "portfolio_allocator"
        
        # Test routing to continue
        sample_state["agent_states"] = {
            "momentum_agent": {"status": "completed"},
            "mean_reversion_agent": {"status": "running"}
        }
        
        route = workflow._route_to_portfolio_or_continue(sample_state)
        assert route == "continue"
    
    def test_risk_routing(self, workflow, sample_state):
        """Test risk-based routing"""
        # Test with high-confidence signals
        sample_state["fused_signals"] = {
            "AAPL": Signal(
                symbol="AAPL",
                signal_type="fused",
                value=0.8,
                confidence=0.9,
                top_3_reasons=["Strong signal", "High confidence", "Good setup"],
                timestamp=datetime.now(),
                model_version="1.0.0",
                agent_name="portfolio_allocator"
            )
        }
        
        route = workflow._route_risk_or_execution(sample_state)
        assert route == "risk_manager"
        
        # Test with no signals
        sample_state["fused_signals"] = {}
        route = workflow._route_risk_or_execution(sample_state)
        assert route == "halt"
    
    def test_monitoring_integration(self, workflow):
        """Test monitoring integration"""
        workflow.enable_monitoring()
        assert workflow.monitoring_enabled is True
        
        workflow.disable_monitoring()
        assert workflow.monitoring_enabled is False


class TestCommunicationProtocols:
    """Test cases for communication protocols"""
    
    @pytest.fixture
    def message_bus(self):
        """Create message bus for testing"""
        return create_message_bus()
    
    @pytest.fixture
    def coordinator(self, message_bus):
        """Create agent coordinator for testing"""
        return create_agent_coordinator(message_bus)
    
    @pytest.fixture
    def conflict_resolver(self):
        """Create conflict resolver for testing"""
        return create_conflict_resolver()
    
    @pytest.mark.asyncio
    async def test_message_bus_operations(self, message_bus):
        """Test message bus publish/subscribe"""
        received_messages = []
        
        def message_handler(message):
            received_messages.append(message)
        
        # Subscribe to messages
        message_bus.subscribe(MessageType.MARKET_DATA_UPDATE, message_handler)
        
        # Start message bus
        await message_bus.start()
        
        # Publish a message
        test_message = Message(
            id="test_1",
            type=MessageType.MARKET_DATA_UPDATE,
            sender="test_agent",
            data={"symbol": "AAPL", "price": 150.0}
        )
        
        await message_bus.publish(test_message)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify message received
        assert len(received_messages) == 1
        assert received_messages[0].type == MessageType.MARKET_DATA_UPDATE
        
        await message_bus.stop()
    
    def test_agent_registration(self, coordinator):
        """Test agent registration with coordinator"""
        coordinator.register_agent(
            "test_agent",
            AgentRole.SIGNAL_GENERATOR,
            {"strategies": ["momentum"]}
        )
        
        status = coordinator.get_agent_status()
        assert status["total_agents"] == 1
        assert "test_agent" in status["agents"]
        assert status["agents"]["test_agent"]["role"] == AgentRole.SIGNAL_GENERATOR
    
    def test_conflict_detection(self, conflict_resolver):
        """Test signal conflict detection"""
        signals = {
            "signal_1": {
                "symbol": "AAPL",
                "value": 0.8,  # Buy signal
                "confidence": 0.9
            },
            "signal_2": {
                "symbol": "AAPL", 
                "value": -0.7,  # Sell signal
                "confidence": 0.8
            }
        }
        
        conflicts = conflict_resolver.detect_conflicts(signals)
        assert len(conflicts) > 0
        assert conflicts[0]["type"] == "opposing_signals"
        assert conflicts[0]["symbol"] == "AAPL"
    
    def test_conflict_resolution(self, conflict_resolver):
        """Test conflict resolution strategies"""
        conflict = {
            "symbol": "AAPL",
            "type": "opposing_signals",
            "signals": [
                ("signal_1", {"value": 0.8, "confidence": 0.9}),
                ("signal_2", {"value": -0.6, "confidence": 0.7})
            ]
        }
        
        # Test weighted average resolution
        resolution = conflict_resolver.resolve_conflict(conflict, "weighted_average")
        assert "resolved_signal" in resolution
        assert resolution["method"] == "weighted_average"
        
        # Test highest confidence resolution
        resolution = conflict_resolver.resolve_conflict(conflict, "highest_confidence")
        assert resolution["resolved_signal"]["confidence"] == 0.9


class TestWorkflowMonitoring:
    """Test cases for workflow monitoring"""
    
    @pytest.fixture
    def monitor(self):
        """Create workflow monitor for testing"""
        return create_workflow_monitor(MonitoringLevel.DEBUG)
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing"""
        return MetricsCollector()
    
    def test_metrics_collection(self, metrics_collector):
        """Test metrics collection"""
        # Record some metrics
        metrics_collector.record_execution_time("test_component", 1.5)
        metrics_collector.record_success("test_component")
        metrics_collector.record_execution_time("test_component", 2.0)
        metrics_collector.record_error("test_component")
        
        # Get component metrics
        metrics = metrics_collector.get_component_metrics("test_component")
        
        assert metrics["component"] == "test_component"
        assert metrics["total_executions"] == 2
        assert metrics["success_rate"] == 0.5
        assert metrics["avg_execution_time"] == 1.75
        assert metrics["error_count"] == 1
    
    @pytest.mark.asyncio
    async def test_monitored_execution(self, monitor):
        """Test monitored execution context manager"""
        with MonitoredExecution(monitor, "test_component"):
            await asyncio.sleep(0.1)  # Simulate work
        
        # Check that execution was recorded
        dashboard = monitor.get_monitoring_dashboard()
        assert "system_metrics" in dashboard
        assert "event_summary" in dashboard
    
    def test_alert_management(self, monitor):
        """Test alert creation and management"""
        from agents.workflow_monitoring import AlertSeverity
        
        # Create an alert
        alert_id = monitor.alert_manager.create_alert(
            "test_component",
            AlertSeverity.WARNING,
            "Test alert message"
        )
        
        # Check alert exists
        active_alerts = monitor.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == alert_id
        
        # Resolve alert
        monitor.alert_manager.resolve_alert(alert_id)
        active_alerts = monitor.alert_manager.get_active_alerts()
        assert len(active_alerts) == 0
    
    def test_debugging_tools(self, monitor):
        """Test debugging functionality"""
        debugger = monitor.debugger
        
        # Set breakpoint
        debugger.set_breakpoint("test_component")
        
        # Check breakpoint
        test_state = {"test_data": "value"}
        should_break = debugger.check_breakpoint("test_component", test_state)
        assert should_break is True
        
        # Test tracing
        debugger.enable_trace()
        debugger.trace_execution("test_component", "test_operation", {"data": "test"})
        
        trace = debugger.get_execution_trace(count=1)
        assert len(trace) == 1
        assert trace[0]["component"] == "test_component"


class TestConditionalRouter:
    """Test cases for conditional routing"""
    
    @pytest.fixture
    def router(self):
        """Create conditional router for testing"""
        return ConditionalRouter()
    
    @pytest.fixture
    def sample_state_with_performance(self):
        """Create sample state with performance data"""
        return TradingSystemState(
            market_data={},
            historical_data={},
            news_articles=[],
            sentiment_scores={},
            market_events=[],
            raw_signals={},
            fused_signals={},
            signal_conflicts=[],
            portfolio_state={},
            positions={},
            risk_metrics=RiskMetrics(
                portfolio_value=100000.0,
                daily_pnl=0.0,
                var_95=0.0,
                max_drawdown=0.0,
                position_count=0,
                leverage=1.0,
                risk_score=0.0
            ),
            risk_limits={},
            pending_orders=[],
            executed_orders=[],
            execution_reports=[],
            market_regime=MarketRegime.HIGH_VOLATILITY,
            workflow_phase=WorkflowPhase.SIGNAL_GENERATION,
            system_alerts=[],
            performance_metrics={},
            symbols_universe=["AAPL"],
            active_strategies=["momentum"],
            model_versions={},
            agent_states={},
            execution_log=[
                {"component": "momentum_agent", "message": "Signal generated successfully"},
                {"component": "momentum_agent", "message": "Signal generated successfully"},
                {"component": "mean_reversion_agent", "message": "Error in signal generation"}
            ],
            error_log=[]
        )
    
    def test_market_regime_routing(self, router, sample_state_with_performance):
        """Test routing based on market regime"""
        routing_decision = router.route_based_on_market_regime(sample_state_with_performance)
        
        assert routing_decision["market_regime"] == MarketRegime.HIGH_VOLATILITY
        assert "momentum" in routing_decision["preferred_strategies"]
        assert routing_decision["risk_adjustment"] == 0.5  # High volatility reduces risk
        assert routing_decision["execution_priority"] == "fast"
    
    def test_strategy_performance_calculation(self, router, sample_state_with_performance):
        """Test strategy performance calculation"""
        performance_scores = router._calculate_strategy_performance(sample_state_with_performance)
        
        # Momentum agent should have higher performance (2/2 success)
        assert performance_scores.get("momentum", 0) > performance_scores.get("mean_reversion", 0)
    
    def test_routing_confidence(self, router, sample_state_with_performance):
        """Test routing confidence calculation"""
        confidence = router._calculate_routing_confidence(sample_state_with_performance)
        
        assert 0.0 <= confidence <= 1.0


class TestWorkflowOrchestrator:
    """Test cases for workflow orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create workflow orchestrator for testing"""
        return create_workflow_orchestrator()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        with patch('agents.communication_protocols.create_message_bus'), \
             patch('agents.communication_protocols.create_agent_coordinator'), \
             patch('agents.workflow_monitoring.create_workflow_monitor'):
            
            await orchestrator.initialize()
            
            status = orchestrator.get_system_status()
            assert "orchestrator" in status
            assert "workflow" in status
    
    def test_agent_role_determination(self, orchestrator):
        """Test agent role determination"""
        role = orchestrator._determine_agent_role("market_data_ingestor")
        assert role == AgentRole.DATA_PROVIDER
        
        role = orchestrator._determine_agent_role("momentum_agent")
        assert role == AgentRole.SIGNAL_GENERATOR
        
        role = orchestrator._determine_agent_role("risk_manager")
        assert role == AgentRole.RISK_MANAGER
    
    def test_agent_capabilities(self, orchestrator):
        """Test agent capabilities retrieval"""
        capabilities = orchestrator._get_agent_capabilities("momentum_agent")
        assert "indicators" in capabilities
        assert "ema" in capabilities["indicators"]
        
        capabilities = orchestrator._get_agent_capabilities("options_agent")
        assert "greeks" in capabilities
        assert capabilities["greeks"] is True


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow execution"""
        # This test would require more complex setup and mocking
        # For now, we'll test the basic integration points
        
        workflow = create_trading_workflow()
        orchestrator = create_workflow_orchestrator()
        
        # Verify components can be created
        assert workflow is not None
        assert orchestrator is not None
        
        # Verify workflow status
        status = workflow.get_workflow_status()
        assert status["graph_compiled"] is True
        assert status["agents_initialized"] > 0
    
    def test_factory_functions(self):
        """Test factory functions"""
        workflow = create_trading_workflow()
        assert isinstance(workflow, LangGraphTradingWorkflow)
        
        orchestrator = create_workflow_orchestrator()
        assert isinstance(orchestrator, WorkflowOrchestrator)
        
        message_bus = create_message_bus()
        assert isinstance(message_bus, MessageBus)
        
        coordinator = create_agent_coordinator(message_bus)
        assert isinstance(coordinator, AgentCoordinator)
        
        resolver = create_conflict_resolver()
        assert isinstance(resolver, ConflictResolver)
        
        monitor = create_workflow_monitor()
        assert isinstance(monitor, WorkflowMonitor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])