# Level 4 AI Trading Agent - API Keys Setup

Your Level 4 AI Trading Agent now supports **multiple LLM providers**! You can use any of these:

## 🥇 **RECOMMENDED: Anthropic (Claude)**
Most accessible and powerful for trading analysis:

1. **Get API Key**: Visit https://console.anthropic.com/
2. **Set Environment Variable**:
   ```bash
   # Windows
   set ANTHROPIC_API_KEY=your_key_here

   # Or add to .env file:
   ANTHROPIC_API_KEY=your_key_here
   ```

## 🥈 **OpenRouter (Multiple Models)**
Access Claude + other models through one API:

1. **Get API Key**: Visit https://openrouter.ai/
2. **Set Environment Variable**:
   ```bash
   # Windows
   set OPENROUTER_API_KEY=your_key_here

   # Or add to .env file:
   OPENROUTER_API_KEY=your_key_here
   ```

## 🥉 **OpenAI (Fallback)**
Traditional option:

1. **Get API Key**: Visit https://platform.openai.com/
2. **Set Environment Variable**:
   ```bash
   # Windows
   set OPENAI_API_KEY=your_key_here

   # Or add to .env file:
   OPENAI_API_KEY=your_key_here
   ```

## 🚀 **Usage Instructions**

### **With API Key (Full Power)**:
```bash
python level4_ai_trading_agent.py
```
- Runs all 9 AI agents
- Neural networks + LLM analysis
- Full autonomous cycle

### **Without API Key (Neural-Only Mode)**:
```bash
python level4_ai_trading_agent.py
```
- Still runs! Uses neural networks only
- PyTorch pattern recognition
- Rule-based regime detection

### **Options Pricing (Always Works)**:
```bash
python options_pricing_integration.py
```
- Black-Scholes options analysis
- Pattern matching to your 68.3% ROI successes
- No API key required

## 💡 **Pro Tips**:

1. **Anthropic Claude** is best for trading analysis
2. **OpenRouter** gives you access to Claude + other models
3. **Neural-only mode** still provides valuable pattern recognition
4. Your successful trading patterns are hardcoded - the system works even without LLMs!

## 🎯 **Priority Setup**:

1. Set `ANTHROPIC_API_KEY` first (recommended)
2. Run `python level4_ai_trading_agent.py`
3. Watch your 68.3% avg ROI patterns get scaled by AI!

The Level 4 AI agent will automatically detect which API key is available and use the best option.