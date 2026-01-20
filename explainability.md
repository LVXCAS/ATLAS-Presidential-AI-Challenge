# Explainability Artifact (ATLAS)

This artifact shows how ATLAS turns offline data into an explainable
risk posture (GREENLIGHT / WATCH / STAND_DOWN).

## Flow diagram (Mermaid)

```mermaid
flowchart TD
    A[Historical CSV or Synthetic Window] --> B[Indicator Proxies
    EMA / RSI / ATR / Bollinger / MACD / ADX]

    subgraph Agents[Interpretable Risk Agents]
      C1[TechnicalAgent
      volatility & momentum]
      C2[MarketRegimeAgent
      trend vs choppy]
      C3[CorrelationAgent
      overlap risk]
      C4[GSQuantAgent
      VaR-style proxy]
      C5[MonteCarloAgent
      tail risk sims]
      C6[RiskManagementAgent
      account limits]
      C7[NewsFilterAgent
      calendar risk]
      C8[SessionTimingAgent
      liquidity timing]
      C9[MultiTimeframeAgent
      trend agreement]
      C10[VolumeLiquidityAgent
      volume/spread proxy]
      C11[SupportResistanceAgent
      key levels]
      C12[DivergenceAgent
      RSI vs price]
      C13[OfflineMLRiskAgent
      offline ML risk lens]
    end

    B --> Agents
    Agents --> D[Agent votes + confidence + reasoning]
    D --> E[Coordinator
    weighted aggregation + veto]
    E --> F[Risk posture + risk flags + explanation]

    E -->|veto if high risk| F
```

## Narrative trace (example)

**Scenario:** A synthetic stress window shows rising volatility and a shift from
calm to choppy price action.

**Trace:**
1. Indicators show elevated ATR (volatility proxy) and low ADX (choppy regime).
2. The TechnicalAgent flags HIGH_VOLATILITY; the MarketRegimeAgent flags
   REGIME_SHIFT; the MonteCarloAgent reports a high probability of large moves.
3. The coordinator aggregates scores and applies the safety veto rule.
4. Output: **STAND_DOWN** with risk flags
   `[HIGH_VOLATILITY, REGIME_SHIFT]` and a plain-language explanation.

**Why this matters:** Students see *why* the system is cautious and learn that
uncertainty is a valid reason to pause.
