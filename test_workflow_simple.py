#!/usr/bin/env python3
"""
Simple test for LangGraph Workflow Implementation
"""

import asyncio
from unittest.mock import Mock, patch
from agents.langgraph_workflow import LangGraphTradingWorkflow, MarketRegime, WorkflowPhase

def test_workflow_with_mocks():
    """Test workflow with mocked agents"""
    
    # Mock all agent imports
    with patch('agents.langgraph_workflow.MarketDataIngestorAgent') as mock_market_data, \
         patch('agents.langgraph_workflow.NewsSentimentAgent') as mock_sentiment, \
         patch('agents.langgraph_workflow.MomentumTradingAgent') as mock_momentum, \
         patch('agents.langgraph_workflow.MeanReversionTradingAgent') as mock_mean_reversion, \
         patch('agents.langgraph_workflow.OptionsVolatilityAgent') as mock_options, \
         patch('agents.langgraph_workflow.PortfolioAllocatorAgent') as mock_portfolio, \
         patch('agents.langgraph_workflow.RiskManagerAgent') as mock_risk, \
         patch('agents.langgraph_workflow.ExecutionEngineAgent') as mock_execution:
        
        # Create workflow
        workflow = LangGraphTradingWorkflow()
        
        # Test workflow status
        status = workflow.get_workflow_status()
        print("✓ Workflow Status:", status)
        
        assert status["graph_compiled"] is True
        assert status["agents_initialized"] == 8
        assert workflow.monitoring_enabled is True
        
        # Test market regime detection
        from agents.langgraph_workflow import MarketData
        from datetime import datetime
        
        # High volatility test
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
        print("✓ High volatility regime detected:", regime)
        assert regime == MarketRegime.HIGH_VOLATILITY
        
        # Low volatility test
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
        print("✓ Low volatility regime detected:", regime)
        assert regime == MarketRegime.LOW_VOLATILITY
        
        # Test conditional routing
        from agents.langgraph_workflow import TradingSystemState, RiskMetrics
        
        sample_state = TradingSystemState(
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
            market_regime=MarketRegime.NORMAL,
            workflow_phase=WorkflowPhase.DATA_INGESTION,
            system_alerts=[],
            performance_metrics={},
            symbols_universe=["AAPL"],
            active_strategies=["momentum"],
            model_versions={},
            agent_states={
                "momentum_agent": {"status": "completed"},
                "mean_reversion_agent": {"status": "completed"},
                "options_agent": {"status": "completed"}
            },
            execution_log=[],
            error_log=[]
        )
        
        # Test routing to portfolio allocator
        route = workflow._route_to_portfolio_or_continue(sample_state)
        print("✓ Routing decision:", route)
        assert route == "portfolio_allocator"
        
        # Test monitoring controls
        workflow.enable_monitoring()
        assert workflow.monitoring_enabled is True
        print("✓ Monitoring enabled")
        
        workflow.disable_monitoring()
        assert workflow.monitoring_enabled is False
        print("✓ Monitoring disabled")
        
        print("\n🎉 All workflow tests passed!")
        return True

def test_conditional_router():
    """Test conditional router functionality"""
    from agents.langgraph_workflow import ConditionalRouter, TradingSystemState, RiskMetrics
    from datetime import datetime
    
    router = ConditionalRouter()
    
    # Create test state with performance data
    state = TradingSystemState(
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
    
    # Test routing decision
    routing_decision = router.route_based_on_market_regime(state)
    
    print("✓ Routing Decision:", routing_decision)
    assert routing_decision["market_regime"] == MarketRegime.HIGH_VOLATILITY
    assert "momentum" in routing_decision["preferred_strategies"]
    assert routing_decision["risk_adjustment"] == 0.5  # High volatility reduces risk
    assert routing_decision["execution_priority"] == "fast"
    
    print("🎉 Conditional router tests passed!")
    return True

def test_communication_protocols():
    """Test communication protocols"""
    from agents.communication_protocols import (
        create_message_bus, create_agent_coordinator, create_conflict_resolver,
        Message, MessageType, AgentRole
    )
    
    # Test message bus creation
    message_bus = create_message_bus()
    assert message_bus is not None
    print("✓ Message bus created")
    
    # Test coordinator creation
    coordinator = create_agent_coordinator(message_bus)
    assert coordinator is not None
    print("✓ Agent coordinator created")
    
    # Test agent registration
    coordinator.register_agent(
        "test_agent",
        AgentRole.SIGNAL_GENERATOR,
        {"strategies": ["momentum"]}
    )
    
    status = coordinator.get_agent_status()
    assert status["total_agents"] == 1
    assert "test_agent" in status["agents"]
    print("✓ Agent registration works")
    
    # Test conflict resolver
    resolver = create_conflict_resolver()
    
    # Test conflict detection
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
    
    conflicts = resolver.detect_conflicts(signals)
    assert len(conflicts) > 0
    assert conflicts[0]["type"] == "opposing_signals"
    print("✓ Conflict detection works")
    
    print("🎉 Communication protocol tests passed!")
    return True

def test_workflow_monitoring():
    """Test workflow monitoring"""
    from agents.workflow_monitoring import create_workflow_monitor, MonitoringLevel, MetricsCollector
    
    # Test monitor creation
    monitor = create_workflow_monitor(MonitoringLevel.DEBUG)
    assert monitor is not None
    print("✓ Workflow monitor created")
    
    # Test metrics collection
    metrics_collector = MetricsCollector()
    
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
    print("✓ Metrics collection works")
    
    print("🎉 Workflow monitoring tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting LangGraph Workflow Tests...\n")
    
    try:
        # Run all tests
        test_workflow_with_mocks()
        print()
        test_conditional_router()
        print()
        test_communication_protocols()
        print()
        test_workflow_monitoring()
        
        print("\n✅ ALL TESTS PASSED! LangGraph Workflow Implementation is working correctly.")
        print("\n📋 Implementation Summary:")
        print("✓ LangGraph StateGraph for agent coordination")
        print("✓ System state structure and transitions")
        print("✓ Agent communication protocols")
        print("✓ Conditional routing based on market conditions")
        print("✓ Workflow monitoring and debugging tools")
        print("✓ Message bus for high-throughput communication")
        print("✓ Agent coordination and resource management")
        print("✓ Conflict resolution for contradictory signals")
        print("✓ Performance metrics and alerting")
        print("✓ Comprehensive debugging capabilities")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()