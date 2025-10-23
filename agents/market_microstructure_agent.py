"""
Market Microstructure Agent - Order Flow and Liquidity Analysis

This agent analyzes market microstructure to optimize trade execution:
- Order book depth and imbalance analysis
- Bid-ask spread monitoring and cost estimation
- Market impact prediction for position sizing
- Volume profile and liquidity analysis
- Best execution timing recommendations
- Slippage prediction and mitigation

The agent helps avoid costly execution by identifying optimal entry/exit timing.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from decimal import Decimal

# LangGraph imports
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


class LiquidityLevel(Enum):
    """Liquidity classification levels"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class MarketImpact(Enum):
    """Market impact severity"""
    MINIMAL = "minimal"      # < 0.1%
    LOW = "low"              # 0.1-0.3%
    MODERATE = "moderate"    # 0.3-0.6%
    HIGH = "high"            # 0.6-1.0%
    SEVERE = "severe"        # > 1.0%


@dataclass
class OrderBookSnapshot:
    """Order book snapshot at a point in time"""
    symbol: str
    timestamp: datetime
    bid_prices: List[float]
    bid_sizes: List[float]
    ask_prices: List[float]
    ask_sizes: List[float]
    mid_price: float
    spread_bps: float

    def total_bid_volume(self, levels: int = 5) -> float:
        """Total bid volume at top N levels"""
        return sum(self.bid_sizes[:levels])

    def total_ask_volume(self, levels: int = 5) -> float:
        """Total ask volume at top N levels"""
        return sum(self.ask_sizes[:levels])

    def imbalance_ratio(self, levels: int = 5) -> float:
        """Order book imbalance ratio (-1 to 1)"""
        bid_vol = self.total_bid_volume(levels)
        ask_vol = self.total_ask_volume(levels)
        total_vol = bid_vol + ask_vol

        if total_vol == 0:
            return 0.0

        return (bid_vol - ask_vol) / total_vol


@dataclass
class VolumeProfile:
    """Volume profile analysis"""
    symbol: str
    timestamp: datetime
    price_levels: List[float]
    volumes: List[float]
    poc_price: float  # Point of Control (highest volume)
    vah_price: float  # Value Area High
    val_price: float  # Value Area Low
    total_volume: float


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity metrics"""
    symbol: str
    timestamp: datetime
    liquidity_level: LiquidityLevel
    spread_bps: float
    depth_score: float  # 0-100
    avg_trade_size: float
    turnover_ratio: float
    days_to_liquidate: float  # For given position size
    kyle_lambda: float  # Price impact coefficient
    amihud_illiquidity: float


@dataclass
class ExecutionRecommendation:
    """Execution timing and strategy recommendation"""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    timing: str  # 'IMMEDIATE', 'PATIENT', 'SPLIT'
    execution_strategy: str  # 'MARKET', 'LIMIT', 'VWAP', 'TWAP', 'ICEBERG'
    estimated_slippage_bps: float
    estimated_impact_bps: float
    confidence: float
    reasoning: List[str]
    recommended_limit_price: Optional[float] = None
    recommended_chunks: Optional[int] = None
    recommended_interval_minutes: Optional[int] = None


@dataclass
class MicrostructureState:
    """LangGraph state for microstructure analysis"""
    symbol: str
    action: str
    quantity: float
    order_book: Optional[OrderBookSnapshot] = None
    volume_profile: Optional[VolumeProfile] = None
    liquidity_metrics: Optional[LiquidityMetrics] = None
    historical_trades: List[Dict[str, Any]] = field(default_factory=list)
    execution_recommendation: Optional[ExecutionRecommendation] = None
    errors: List[str] = field(default_factory=list)


class MarketMicrostructureAgent:
    """
    Market Microstructure Agent for optimal trade execution

    Analyzes order flow, liquidity, and market impact to determine:
    - Best execution timing
    - Optimal order type and size
    - Expected slippage and market impact
    - Liquidity conditions
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Market Microstructure Agent"""
        self.config = config or {}

        # Configuration parameters
        self.order_book_levels = self.config.get('order_book_levels', 10)
        self.volume_profile_bins = self.config.get('volume_profile_bins', 50)
        self.max_impact_bps = self.config.get('max_impact_bps', 50)  # 0.5%
        self.max_position_adv_pct = self.config.get('max_position_adv_pct', 10)  # 10% of ADV

        # Kyle's Lambda (price impact) parameters
        self.kyle_lambda_default = self.config.get('kyle_lambda_default', 0.0001)

        # Build workflow
        self.workflow = self._create_workflow()

        logger.info("Market Microstructure Agent initialized")

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for microstructure analysis"""
        workflow = StateGraph(MicrostructureState)

        # Add nodes
        workflow.add_node("fetch_order_book", self._fetch_order_book)
        workflow.add_node("analyze_liquidity", self._analyze_liquidity)
        workflow.add_node("build_volume_profile", self._build_volume_profile)
        workflow.add_node("calculate_market_impact", self._calculate_market_impact)
        workflow.add_node("generate_execution_plan", self._generate_execution_plan)

        # Define edges
        workflow.set_entry_point("fetch_order_book")
        workflow.add_edge("fetch_order_book", "analyze_liquidity")
        workflow.add_edge("analyze_liquidity", "build_volume_profile")
        workflow.add_edge("build_volume_profile", "calculate_market_impact")
        workflow.add_edge("calculate_market_impact", "generate_execution_plan")
        workflow.add_edge("generate_execution_plan", END)

        return workflow.compile()

    async def analyze_execution(self, symbol: str, action: str, quantity: float) -> ExecutionRecommendation:
        """
        Main entry point: analyze optimal execution for a trade

        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares/contracts

        Returns:
            ExecutionRecommendation with timing and strategy
        """
        try:
            initial_state = MicrostructureState(
                symbol=symbol,
                action=action,
                quantity=quantity
            )

            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)

            if final_state.execution_recommendation:
                return final_state.execution_recommendation
            else:
                # Fallback recommendation
                return self._create_fallback_recommendation(symbol, action, quantity)

        except Exception as e:
            logger.error(f"Error in microstructure analysis for {symbol}: {e}")
            return self._create_fallback_recommendation(symbol, action, quantity)

    async def _fetch_order_book(self, state: MicrostructureState) -> MicrostructureState:
        """Fetch current order book snapshot"""
        try:
            symbol = state.symbol

            # ENHANCEMENT: Fetch real recent market data for better order book simulation
            from agents.real_data_connector import fetch_real_market_data, get_current_price

            logger.info(f"Fetching REAL market data for order book construction: {symbol}")

            # Fetch recent intraday data (last 10 days, 5-min bars for microstructure analysis)
            recent_data = await fetch_real_market_data(symbol, days=10, timeframe="5Min")

            if recent_data is not None and len(recent_data) > 0:
                logger.info(f"✅ Using REAL data: {len(recent_data)} intraday bars for order book")
                # Get current price from real data
                current_price = recent_data['close'].iloc[-1]
                # Simulate order book using real price levels
                order_book = await self._simulate_order_book(symbol, real_price=current_price, real_data=recent_data)
            else:
                logger.warning(f"Real intraday data unavailable, falling back to daily data")
                # Try daily data as fallback
                daily_data = await fetch_real_market_data(symbol, days=10, timeframe="1Day")
                if daily_data is not None and len(daily_data) > 0:
                    current_price = daily_data['close'].iloc[-1]
                    order_book = await self._simulate_order_book(symbol, real_price=current_price, real_data=daily_data)
                else:
                    # Final fallback: simulate without real data
                    logger.warning(f"No real data available, using pure simulation")
                    order_book = await self._simulate_order_book(symbol)

            state.order_book = order_book
            logger.info(f"Order book fetched for {symbol}: spread={order_book.spread_bps:.2f} bps")

        except Exception as e:
            logger.error(f"Error fetching order book for {state.symbol}: {e}")
            # Emergency fallback
            try:
                order_book = await self._simulate_order_book(state.symbol)
                state.order_book = order_book
            except:
                state.errors.append(f"Order book fetch failed: {str(e)}")

        return state

    async def _analyze_liquidity(self, state: MicrostructureState) -> MicrostructureState:
        """Analyze liquidity metrics"""
        try:
            if not state.order_book:
                state.errors.append("No order book data available")
                return state

            ob = state.order_book

            # Calculate depth score (0-100)
            bid_depth = ob.total_bid_volume(5)
            ask_depth = ob.total_ask_volume(5)
            total_depth = bid_depth + ask_depth
            depth_score = min(100, total_depth / 1000 * 100)  # Normalize

            # Estimate average trade size (simplified)
            avg_trade_size = total_depth / 10

            # Estimate days to liquidate for position
            avg_daily_volume = await self._get_average_daily_volume(state.symbol)
            max_participation_rate = 0.1  # Don't exceed 10% of daily volume
            days_to_liquidate = state.quantity / (avg_daily_volume * max_participation_rate) if avg_daily_volume > 0 else 999

            # Kyle's Lambda (price impact coefficient)
            kyle_lambda = self._estimate_kyle_lambda(ob, avg_daily_volume)

            # Amihud illiquidity ratio
            amihud_illiquidity = self._calculate_amihud_illiquidity(state.symbol)

            # Classify liquidity level
            liquidity_level = self._classify_liquidity(
                spread_bps=ob.spread_bps,
                depth_score=depth_score,
                days_to_liquidate=days_to_liquidate
            )

            # Estimate turnover ratio
            market_cap = await self._get_market_cap(state.symbol)
            turnover_ratio = avg_daily_volume / market_cap if market_cap > 0 else 0

            state.liquidity_metrics = LiquidityMetrics(
                symbol=state.symbol,
                timestamp=datetime.now(timezone.utc),
                liquidity_level=liquidity_level,
                spread_bps=ob.spread_bps,
                depth_score=depth_score,
                avg_trade_size=avg_trade_size,
                turnover_ratio=turnover_ratio,
                days_to_liquidate=days_to_liquidate,
                kyle_lambda=kyle_lambda,
                amihud_illiquidity=amihud_illiquidity
            )

            logger.info(f"Liquidity analysis for {state.symbol}: {liquidity_level.value}, depth={depth_score:.1f}")

        except Exception as e:
            logger.error(f"Error analyzing liquidity: {e}")
            state.errors.append(f"Liquidity analysis failed: {str(e)}")

        return state

    async def _build_volume_profile(self, state: MicrostructureState) -> MicrostructureState:
        """Build volume profile from recent trades"""
        try:
            # Fetch recent trades
            trades = await self._fetch_recent_trades(state.symbol, hours=24)

            if not trades:
                state.errors.append("No trade data for volume profile")
                return state

            # Build volume profile
            df = pd.DataFrame(trades)

            # Create price bins
            price_min = df['price'].min()
            price_max = df['price'].max()
            bins = np.linspace(price_min, price_max, self.volume_profile_bins)

            # Aggregate volume by price level
            df['price_bin'] = pd.cut(df['price'], bins=bins)
            volume_by_price = df.groupby('price_bin')['size'].sum()

            # Find Point of Control (POC) - highest volume price
            poc_idx = volume_by_price.idxmax()
            poc_price = (poc_idx.left + poc_idx.right) / 2

            # Calculate Value Area (70% of volume)
            total_volume = volume_by_price.sum()
            target_volume = total_volume * 0.70

            # Find value area high and low
            sorted_volumes = volume_by_price.sort_values(ascending=False)
            cumulative_volume = 0
            value_area_bins = []

            for bin_range, volume in sorted_volumes.items():
                cumulative_volume += volume
                value_area_bins.append(bin_range)
                if cumulative_volume >= target_volume:
                    break

            # Extract value area prices
            value_area_prices = [(b.left + b.right) / 2 for b in value_area_bins]
            vah_price = max(value_area_prices)
            val_price = min(value_area_prices)

            state.volume_profile = VolumeProfile(
                symbol=state.symbol,
                timestamp=datetime.now(timezone.utc),
                price_levels=[float((b.left + b.right) / 2) for b in volume_by_price.index],
                volumes=volume_by_price.values.tolist(),
                poc_price=poc_price,
                vah_price=vah_price,
                val_price=val_price,
                total_volume=float(total_volume)
            )

            logger.info(f"Volume profile built for {state.symbol}: POC={poc_price:.2f}")

        except Exception as e:
            logger.error(f"Error building volume profile: {e}")
            state.errors.append(f"Volume profile failed: {str(e)}")

        return state

    async def _calculate_market_impact(self, state: MicrostructureState) -> MicrostructureState:
        """Calculate expected market impact of the trade"""
        try:
            if not state.liquidity_metrics or not state.order_book:
                return state

            metrics = state.liquidity_metrics
            ob = state.order_book

            # Permanent impact (Kyle's Lambda model)
            # Price impact = lambda * (quantity / sqrt(daily_volume))
            avg_daily_volume = await self._get_average_daily_volume(state.symbol)
            permanent_impact_bps = (
                metrics.kyle_lambda * state.quantity / np.sqrt(avg_daily_volume) * 10000
            ) if avg_daily_volume > 0 else 0

            # Temporary impact (from spread and depth)
            spread_cost_bps = ob.spread_bps / 2  # Half spread

            # Depth-based impact
            if state.action == 'BUY':
                available_liquidity = ob.total_ask_volume(self.order_book_levels)
            else:
                available_liquidity = ob.total_bid_volume(self.order_book_levels)

            liquidity_ratio = state.quantity / available_liquidity if available_liquidity > 0 else 999
            depth_impact_bps = min(100, liquidity_ratio * 20)  # 5% quantity = 1 bps impact

            # Total estimated slippage
            total_slippage_bps = spread_cost_bps + depth_impact_bps

            # Total market impact (permanent + temporary)
            total_impact_bps = permanent_impact_bps + total_slippage_bps

            # Store in state for use in execution plan
            state.estimated_slippage = total_slippage_bps
            state.estimated_impact = total_impact_bps

            logger.info(f"Market impact for {state.symbol}: slippage={total_slippage_bps:.2f} bps, total={total_impact_bps:.2f} bps")

        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            state.errors.append(f"Market impact calculation failed: {str(e)}")

        return state

    async def _generate_execution_plan(self, state: MicrostructureState) -> MicrostructureState:
        """Generate optimal execution recommendation"""
        try:
            if not state.liquidity_metrics or not state.order_book:
                return state

            metrics = state.liquidity_metrics
            ob = state.order_book
            slippage_bps = getattr(state, 'estimated_slippage', 10)
            impact_bps = getattr(state, 'estimated_impact', 20)

            reasoning = []

            # Determine timing
            if metrics.liquidity_level in [LiquidityLevel.VERY_HIGH, LiquidityLevel.HIGH]:
                timing = "IMMEDIATE"
                reasoning.append(f"High liquidity ({metrics.liquidity_level.value})")
            elif slippage_bps > self.max_impact_bps:
                timing = "SPLIT"
                reasoning.append(f"High slippage ({slippage_bps:.1f} bps) - split order")
            elif metrics.days_to_liquidate > 1:
                timing = "PATIENT"
                reasoning.append(f"Large position ({metrics.days_to_liquidate:.1f} days to liquidate)")
            else:
                timing = "IMMEDIATE"
                reasoning.append("Moderate impact, can execute immediately")

            # Determine execution strategy
            if timing == "IMMEDIATE" and slippage_bps < 10:
                execution_strategy = "MARKET"
                reasoning.append("Low slippage allows market order")
            elif timing == "IMMEDIATE":
                execution_strategy = "LIMIT"
                # Place limit at mid-price for passive fill
                recommended_limit_price = ob.mid_price
                reasoning.append(f"Limit order at mid-price ({recommended_limit_price:.2f})")
            elif timing == "SPLIT":
                execution_strategy = "TWAP"  # Time-Weighted Average Price
                # Recommend splitting into chunks
                chunks = min(10, int(metrics.days_to_liquidate * 10))
                recommended_chunks = chunks
                recommended_interval_minutes = 5
                reasoning.append(f"Split into {chunks} orders over {chunks * 5} minutes")
            else:  # PATIENT
                execution_strategy = "VWAP"  # Volume-Weighted Average Price
                reasoning.append("Execute over the day matching volume pattern")

            # Calculate recommended limit price for limit orders
            if execution_strategy == "LIMIT" and not hasattr(state, 'recommended_limit_price'):
                if state.action == "BUY":
                    recommended_limit_price = ob.mid_price  # Buy at mid or below
                else:
                    recommended_limit_price = ob.mid_price  # Sell at mid or above
            else:
                recommended_limit_price = getattr(state, 'recommended_limit_price', None)

            # Add liquidity insights
            if metrics.liquidity_level == LiquidityLevel.VERY_LOW:
                reasoning.append("⚠️ VERY LOW liquidity - consider reducing size")

            imbalance = ob.imbalance_ratio(5)
            if abs(imbalance) > 0.3:
                direction = "bullish" if imbalance > 0 else "bearish"
                reasoning.append(f"Order book {direction} (imbalance: {imbalance:.2f})")

            # Check volume profile
            if state.volume_profile:
                vp = state.volume_profile
                current_price = ob.mid_price

                if vp.val_price <= current_price <= vp.vah_price:
                    reasoning.append("Price in value area - good entry zone")
                elif current_price < vp.val_price:
                    reasoning.append("Price below value area - potential support")
                else:
                    reasoning.append("Price above value area - potential resistance")

            # Calculate confidence
            confidence_factors = []

            # Liquidity confidence
            if metrics.liquidity_level in [LiquidityLevel.VERY_HIGH, LiquidityLevel.HIGH]:
                confidence_factors.append(0.9)
            elif metrics.liquidity_level == LiquidityLevel.MODERATE:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)

            # Impact confidence
            if impact_bps < 20:
                confidence_factors.append(0.9)
            elif impact_bps < 50:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)

            # Spread confidence
            if ob.spread_bps < 10:
                confidence_factors.append(0.9)
            elif ob.spread_bps < 30:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)

            confidence = np.mean(confidence_factors)

            # Create recommendation
            state.execution_recommendation = ExecutionRecommendation(
                symbol=state.symbol,
                action=state.action,
                quantity=state.quantity,
                timing=timing,
                execution_strategy=execution_strategy,
                estimated_slippage_bps=slippage_bps,
                estimated_impact_bps=impact_bps,
                confidence=confidence,
                reasoning=reasoning,
                recommended_limit_price=recommended_limit_price,
                recommended_chunks=getattr(state, 'recommended_chunks', None),
                recommended_interval_minutes=getattr(state, 'recommended_interval_minutes', None)
            )

            logger.info(f"Execution plan for {state.symbol}: {timing} / {execution_strategy} (confidence: {confidence:.2%})")

        except Exception as e:
            logger.error(f"Error generating execution plan: {e}")
            state.errors.append(f"Execution plan failed: {str(e)}")

        return state

    # Helper methods

    async def _simulate_order_book(
        self,
        symbol: str,
        real_price: Optional[float] = None,
        real_data: Optional[pd.DataFrame] = None
    ) -> OrderBookSnapshot:
        """Simulate order book using real price data when available"""

        # ENHANCEMENT: Use real price if available
        if real_price is not None:
            mid_price = float(real_price)
            logger.debug(f"Using REAL mid price: ${mid_price:.2f}")

            # Calculate realistic spread based on recent volatility
            if real_data is not None and len(real_data) > 0:
                # Calculate recent volatility for spread estimation
                returns = real_data['close'].pct_change().dropna()
                volatility = returns.std()
                # Spread typically 2-10 bps, higher for volatile stocks
                spread_pct = min(0.10, max(0.02, volatility * 50))  # 2-10 bps
            else:
                spread_pct = 0.05  # Default 5 bps
        else:
            # Fallback to default simulation
            mid_price = 150.0  # Example price
            spread_pct = 0.05  # 5 bps

        spread_dollars = mid_price * spread_pct / 100

        bid_prices = [mid_price - spread_dollars/2 - i*0.01 for i in range(10)]
        ask_prices = [mid_price + spread_dollars/2 + i*0.01 for i in range(10)]

        bid_sizes = [1000 - i*50 for i in range(10)]  # Decreasing size
        ask_sizes = [1000 - i*50 for i in range(10)]

        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
            mid_price=mid_price,
            spread_bps=spread_pct
        )

    async def _get_average_daily_volume(self, symbol: str) -> float:
        """Get 30-day average daily volume"""
        # Placeholder - fetch from market data
        return 1_000_000  # 1M shares/day

    async def _get_market_cap(self, symbol: str) -> float:
        """Get market capitalization"""
        # Placeholder - fetch from market data
        return 10_000_000_000  # $10B

    async def _fetch_recent_trades(self, symbol: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Fetch recent trades for volume profile"""
        # Placeholder - fetch real trade data
        # Return list of {price, size, timestamp}
        return [
            {'price': 150.0 + np.random.randn()*0.5, 'size': 100 + np.random.randint(0, 500), 'timestamp': datetime.now()}
            for _ in range(1000)
        ]

    def _estimate_kyle_lambda(self, order_book: OrderBookSnapshot, avg_daily_volume: float) -> float:
        """Estimate Kyle's Lambda (price impact coefficient)"""
        # Simplified estimation - in practice, calibrate from historical data
        if avg_daily_volume == 0:
            return self.kyle_lambda_default

        # Lambda roughly inversely proportional to liquidity
        liquidity_factor = order_book.total_bid_volume(5) + order_book.total_ask_volume(5)
        estimated_lambda = 100 / liquidity_factor if liquidity_factor > 0 else self.kyle_lambda_default

        return estimated_lambda

    def _calculate_amihud_illiquidity(self, symbol: str) -> float:
        """Calculate Amihud illiquidity ratio"""
        # Amihud = Average(|Daily Return| / Daily Dollar Volume)
        # Placeholder - calculate from historical data
        return 0.001

    def _classify_liquidity(self, spread_bps: float, depth_score: float, days_to_liquidate: float) -> LiquidityLevel:
        """Classify overall liquidity level"""
        if spread_bps < 5 and depth_score > 80 and days_to_liquidate < 0.5:
            return LiquidityLevel.VERY_HIGH
        elif spread_bps < 10 and depth_score > 60 and days_to_liquidate < 1:
            return LiquidityLevel.HIGH
        elif spread_bps < 20 and depth_score > 40 and days_to_liquidate < 2:
            return LiquidityLevel.MODERATE
        elif spread_bps < 50 and depth_score > 20 and days_to_liquidate < 5:
            return LiquidityLevel.LOW
        else:
            return LiquidityLevel.VERY_LOW

    def _create_fallback_recommendation(self, symbol: str, action: str, quantity: float) -> ExecutionRecommendation:
        """Create safe fallback recommendation when analysis fails"""
        return ExecutionRecommendation(
            symbol=symbol,
            action=action,
            quantity=quantity,
            timing="PATIENT",
            execution_strategy="LIMIT",
            estimated_slippage_bps=20.0,
            estimated_impact_bps=30.0,
            confidence=0.5,
            reasoning=["Analysis incomplete - using conservative defaults"],
            recommended_limit_price=None
        )


# Factory function
def create_market_microstructure_agent(config: Dict[str, Any] = None) -> MarketMicrostructureAgent:
    """Create Market Microstructure Agent instance"""
    return MarketMicrostructureAgent(config)


# Example usage
if __name__ == "__main__":
    async def test_agent():
        agent = create_market_microstructure_agent()

        # Test execution analysis
        recommendation = await agent.analyze_execution(
            symbol="AAPL",
            action="BUY",
            quantity=10000
        )

        print(f"\n=== Execution Recommendation for AAPL ===")
        print(f"Timing: {recommendation.timing}")
        print(f"Strategy: {recommendation.execution_strategy}")
        print(f"Estimated Slippage: {recommendation.estimated_slippage_bps:.2f} bps")
        print(f"Estimated Impact: {recommendation.estimated_impact_bps:.2f} bps")
        print(f"Confidence: {recommendation.confidence:.1%}")
        print(f"\nReasoning:")
        for i, reason in enumerate(recommendation.reasoning, 1):
            print(f"  {i}. {reason}")

        if recommendation.recommended_limit_price:
            print(f"\nRecommended Limit Price: ${recommendation.recommended_limit_price:.2f}")

        if recommendation.recommended_chunks:
            print(f"Split into {recommendation.recommended_chunks} chunks over {recommendation.recommended_interval_minutes} min intervals")

    asyncio.run(test_agent())
