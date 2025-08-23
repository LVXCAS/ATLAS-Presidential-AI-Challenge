#!/usr/bin/env python3
"""
Hive Trade - Advanced NLP System Demonstration
Shows the comprehensive NLP analysis capabilities for news sentiment
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

# Mock news articles for demonstration
SAMPLE_NEWS_ARTICLES = [
    {
        "id": 1,
        "symbol": "AAPL",
        "title": "Apple Reports Record Q4 Earnings, Beats Wall Street Expectations by 12%",
        "content": """Apple Inc. announced today that it has achieved record-breaking quarterly earnings of $94.9 billion, 
        surpassing analyst expectations by a significant margin. The tech giant's strong performance was driven by 
        robust iPhone sales and growing services revenue. CEO Tim Cook stated that the company expects continued 
        growth in the upcoming quarter, with new product launches planned for 2024. Apple's stock price target 
        has been raised to $200 by Goldman Sachs analysts following this exceptional performance.""",
        "source": "Reuters",
        "timestamp": "2024-01-15 16:30:00",
        "category": "earnings"
    },
    {
        "id": 2,
        "symbol": "TSLA",
        "title": "Tesla Faces Production Challenges at Gigafactory Shanghai",
        "content": """Tesla Motors is experiencing significant production delays at its Shanghai Gigafactory due to 
        supply chain disruptions and regulatory compliance issues. The electric vehicle manufacturer has 
        temporarily reduced production capacity by 30% and may miss its Q1 delivery targets. Industry analysts 
        are concerned about Tesla's ability to maintain its growth trajectory. Morgan Stanley has downgraded 
        TSLA to 'Underweight' with a price target of $180, citing operational challenges and increasing 
        competition in the EV market.""",
        "source": "Bloomberg",
        "timestamp": "2024-01-15 14:22:00",
        "category": "operations"
    },
    {
        "id": 3,
        "symbol": "MSFT",
        "title": "Microsoft Azure Cloud Revenue Soars 35% as AI Adoption Accelerates",
        "content": """Microsoft Corporation reported exceptional growth in its Azure cloud computing division, with 
        revenue increasing 35% year-over-year. The surge is attributed to accelerated enterprise adoption of 
        artificial intelligence services and cloud migration initiatives. Microsoft's partnership with OpenAI 
        continues to drive innovation in the AI space. The company announced new AI-powered features for Office 365 
        and plans to invest an additional $10 billion in AI infrastructure this year. Analysts remain bullish 
        on Microsoft's long-term prospects in the cloud and AI markets.""",
        "source": "Financial Times",
        "timestamp": "2024-01-15 15:45:00",
        "category": "technology"
    },
    {
        "id": 4,
        "symbol": "SPY",
        "title": "Federal Reserve Hints at Potential Rate Cuts Amid Cooling Inflation",
        "content": """Federal Reserve Chairman Jerome Powell indicated today that the central bank is considering 
        interest rate cuts in the coming months as inflation continues to moderate. The latest CPI data shows 
        inflation falling to 2.8%, approaching the Fed's 2% target. Powell emphasized that any rate decisions 
        will be data-dependent and cautioned against premature celebrations. Market participants are pricing 
        in a 75% probability of a 25 basis point cut at the next FOMC meeting. This dovish stance has boosted 
        equity markets, with the S&P 500 reaching new all-time highs.""",
        "source": "MarketWatch",
        "timestamp": "2024-01-15 18:15:00",
        "category": "monetary_policy"
    }
]

class MockAdvancedNLPAgent:
    """Mock version of the advanced NLP agent for demonstration"""
    
    def __init__(self):
        # Simulate model loading
        self.sentiment_models = ["FinBERT", "VADER", "Gemini"]
        self.ner_model = "spaCy"
        self.topic_model = "LDA"
        print("Advanced NLP Agent initialized with:")
        print(f"  - Sentiment models: {', '.join(self.sentiment_models)}")
        print(f"  - NER model: {self.ner_model}")
        print(f"  - Topic model: {self.topic_model}")
        print()
    
    async def analyze_sentiment(self, text: str, symbol: str) -> Dict[str, Any]:
        """Simulate comprehensive sentiment analysis"""
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Mock sentiment scores based on text content
        positive_words = ["record", "beats", "strong", "growth", "exceptional", "soars", "bullish", "boost"]
        negative_words = ["challenges", "delays", "disruptions", "miss", "concerned", "downgraded", "reduced"]
        neutral_words = ["announced", "reported", "indicated", "data", "meeting"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        neu_count = sum(1 for word in neutral_words if word in text_lower)
        
        # Calculate composite sentiment
        total_sentiment_words = pos_count + neg_count + neu_count
        if total_sentiment_words == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (pos_count - neg_count) / max(total_sentiment_words, 1)
        
        # Normalize to -1 to 1 range
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        return {
            "finbert_sentiment": {
                "positive": max(0, sentiment_score),
                "negative": max(0, -sentiment_score),
                "neutral": 1 - abs(sentiment_score),
                "score": sentiment_score
            },
            "vader_sentiment": {
                "compound": sentiment_score * 0.8,
                "pos": max(0, sentiment_score * 0.7),
                "neu": 0.6 + abs(sentiment_score) * 0.2,
                "neg": max(0, -sentiment_score * 0.7)
            },
            "composite_sentiment": sentiment_score,
            "confidence": min(0.95, 0.6 + abs(sentiment_score) * 0.35)
        }
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Simulate named entity recognition"""
        await asyncio.sleep(0.3)
        
        entities = []
        
        # Mock entity extraction
        if "Apple" in text or "AAPL" in text:
            entities.append({"text": "Apple Inc.", "label": "ORG", "confidence": 0.95})
        if "Tesla" in text or "TSLA" in text:
            entities.append({"text": "Tesla Motors", "label": "ORG", "confidence": 0.93})
        if "Microsoft" in text or "MSFT" in text:
            entities.append({"text": "Microsoft Corporation", "label": "ORG", "confidence": 0.96})
        if "Goldman Sachs" in text:
            entities.append({"text": "Goldman Sachs", "label": "ORG", "confidence": 0.89})
        if "Federal Reserve" in text:
            entities.append({"text": "Federal Reserve", "label": "ORG", "confidence": 0.92})
        if "Jerome Powell" in text:
            entities.append({"text": "Jerome Powell", "label": "PERSON", "confidence": 0.88})
        if "Tim Cook" in text:
            entities.append({"text": "Tim Cook", "label": "PERSON", "confidence": 0.91})
        
        # Extract financial metrics
        import re
        
        # Price targets
        price_targets = re.findall(r'price target.*?\$(\d+)', text, re.IGNORECASE)
        for target in price_targets:
            entities.append({"text": f"${target}", "label": "PRICE_TARGET", "confidence": 0.85})
        
        # Revenue/earnings figures
        revenue_figures = re.findall(r'\$(\d+(?:\.\d+)?)\s*billion', text, re.IGNORECASE)
        for figure in revenue_figures:
            entities.append({"text": f"${figure}B", "label": "REVENUE", "confidence": 0.82})
        
        return entities
    
    async def detect_market_signals(self, text: str, symbol: str) -> Dict[str, Any]:
        """Simulate market signal detection"""
        await asyncio.sleep(0.2)
        
        signals = {
            "earnings_signals": [],
            "rating_changes": [],
            "price_targets": [],
            "guidance_updates": [],
            "insider_activity": []
        }
        
        text_lower = text.lower()
        
        # Earnings signals
        if "beats" in text_lower and "expectations" in text_lower:
            signals["earnings_signals"].append({
                "type": "earnings_beat",
                "confidence": 0.91,
                "impact": "positive",
                "magnitude": "high"
            })
        elif "misses" in text_lower and "estimates" in text_lower:
            signals["earnings_signals"].append({
                "type": "earnings_miss",
                "confidence": 0.89,
                "impact": "negative",
                "magnitude": "high"
            })
        
        # Rating changes
        if "downgraded" in text_lower or "underweight" in text_lower:
            signals["rating_changes"].append({
                "type": "downgrade",
                "confidence": 0.87,
                "impact": "negative",
                "magnitude": "medium"
            })
        elif "upgraded" in text_lower or "overweight" in text_lower:
            signals["rating_changes"].append({
                "type": "upgrade",
                "confidence": 0.85,
                "impact": "positive",
                "magnitude": "medium"
            })
        
        # Price targets
        import re
        price_targets = re.findall(r'price target.*?\$(\d+)', text, re.IGNORECASE)
        for target in price_targets:
            signals["price_targets"].append({
                "target": int(target),
                "confidence": 0.82,
                "source": "analyst"
            })
        
        # Guidance updates
        if "expects" in text_lower and "growth" in text_lower:
            signals["guidance_updates"].append({
                "type": "positive_guidance",
                "confidence": 0.78,
                "impact": "positive",
                "magnitude": "medium"
            })
        elif "challenges" in text_lower and "targets" in text_lower:
            signals["guidance_updates"].append({
                "type": "guidance_revision",
                "confidence": 0.76,
                "impact": "negative",
                "magnitude": "medium"
            })
        
        return signals
    
    async def calculate_trading_impact(self, sentiment_score: float, signals: Dict, symbol: str) -> Dict[str, Any]:
        """Calculate potential trading impact based on NLP analysis"""
        
        # Base impact from sentiment
        base_impact = sentiment_score * 0.6
        
        # Adjust for signals
        signal_impact = 0.0
        
        for earnings_signal in signals.get("earnings_signals", []):
            if earnings_signal["impact"] == "positive":
                signal_impact += 0.3 * earnings_signal["confidence"]
            else:
                signal_impact -= 0.3 * earnings_signal["confidence"]
        
        for rating_change in signals.get("rating_changes", []):
            if rating_change["impact"] == "positive":
                signal_impact += 0.2 * rating_change["confidence"]
            else:
                signal_impact -= 0.2 * rating_change["confidence"]
        
        for guidance in signals.get("guidance_updates", []):
            if guidance["impact"] == "positive":
                signal_impact += 0.15 * guidance["confidence"]
            else:
                signal_impact -= 0.15 * guidance["confidence"]
        
        total_impact = base_impact + signal_impact
        total_impact = max(-1.0, min(1.0, total_impact))  # Clamp to [-1, 1]
        
        # Generate trading recommendation
        if total_impact > 0.3:
            action = "BUY"
            confidence = min(0.95, 0.6 + total_impact * 0.35)
        elif total_impact < -0.3:
            action = "SELL"
            confidence = min(0.95, 0.6 + abs(total_impact) * 0.35)
        else:
            action = "HOLD"
            confidence = 0.5 + abs(total_impact) * 0.2
        
        return {
            "total_impact_score": total_impact,
            "recommended_action": action,
            "confidence": confidence,
            "position_size_multiplier": min(2.0, 1.0 + abs(total_impact)),
            "risk_adjustment": 1.0 - abs(total_impact) * 0.2,
            "time_horizon": "short" if abs(total_impact) > 0.5 else "medium"
        }

async def demonstrate_nlp_capabilities():
    """Main demonstration function"""
    
    print("=" * 80)
    print("HIVE TRADE - ADVANCED NLP SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize NLP agent
    nlp_agent = MockAdvancedNLPAgent()
    
    analysis_results = []
    
    for i, article in enumerate(SAMPLE_NEWS_ARTICLES, 1):
        print(f"ANALYZING ARTICLE {i}/4: {article['symbol']}")
        print("-" * 50)
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']} | Category: {article['category']}")
        print(f"Timestamp: {article['timestamp']}")
        print()
        
        # Step 1: Sentiment Analysis
        print("Step 1: Comprehensive Sentiment Analysis")
        sentiment_result = await nlp_agent.analyze_sentiment(article['content'], article['symbol'])
        
        print(f"  FinBERT Sentiment: {sentiment_result['finbert_sentiment']['score']:.3f}")
        print(f"    - Positive: {sentiment_result['finbert_sentiment']['positive']:.2f}")
        print(f"    - Negative: {sentiment_result['finbert_sentiment']['negative']:.2f}")
        print(f"    - Neutral: {sentiment_result['finbert_sentiment']['neutral']:.2f}")
        print()
        print(f"  VADER Sentiment: {sentiment_result['vader_sentiment']['compound']:.3f}")
        print(f"    - Positive: {sentiment_result['vader_sentiment']['pos']:.2f}")
        print(f"    - Negative: {sentiment_result['vader_sentiment']['neg']:.2f}")
        print(f"    - Neutral: {sentiment_result['vader_sentiment']['neu']:.2f}")
        print()
        print(f"  Composite Sentiment: {sentiment_result['composite_sentiment']:.3f}")
        print(f"  Confidence: {sentiment_result['confidence']:.2f}")
        print()
        
        # Step 2: Named Entity Recognition
        print("Step 2: Named Entity Recognition")
        entities = await nlp_agent.extract_entities(article['content'])
        
        for entity in entities:
            print(f"  - {entity['text']} ({entity['label']}) - Confidence: {entity['confidence']:.2f}")
        print()
        
        # Step 3: Market Signal Detection
        print("Step 3: Market Signal Detection")
        signals = await nlp_agent.detect_market_signals(article['content'], article['symbol'])
        
        for signal_type, signal_list in signals.items():
            if signal_list:
                print(f"  {signal_type.replace('_', ' ').title()}:")
                for signal in signal_list:
                    if signal_type == "price_targets":
                        print(f"    - Target: ${signal['target']} (Confidence: {signal['confidence']:.2f})")
                    else:
                        print(f"    - {signal['type']}: {signal.get('impact', 'N/A')} impact (Confidence: {signal['confidence']:.2f})")
        print()
        
        # Step 4: Trading Impact Calculation
        print("Step 4: Trading Impact Assessment")
        trading_impact = await nlp_agent.calculate_trading_impact(
            sentiment_result['composite_sentiment'], 
            signals, 
            article['symbol']
        )
        
        print(f"  Total Impact Score: {trading_impact['total_impact_score']:.3f}")
        print(f"  Recommended Action: {trading_impact['recommended_action']}")
        print(f"  Confidence: {trading_impact['confidence']:.2f}")
        print(f"  Position Size Multiplier: {trading_impact['position_size_multiplier']:.2f}x")
        print(f"  Risk Adjustment: {trading_impact['risk_adjustment']:.2f}")
        print(f"  Time Horizon: {trading_impact['time_horizon']}")
        print()
        
        # Store results
        article_analysis = {
            "article": article,
            "sentiment": sentiment_result,
            "entities": entities,
            "signals": signals,
            "trading_impact": trading_impact
        }
        analysis_results.append(article_analysis)
        
        print("=" * 50)
        print()
    
    # Generate Summary Report
    print("NLP ANALYSIS SUMMARY REPORT")
    print("=" * 80)
    print()
    
    print("TRADING RECOMMENDATIONS:")
    print("-" * 30)
    
    for result in analysis_results:
        symbol = result['article']['symbol']
        action = result['trading_impact']['recommended_action']
        confidence = result['trading_impact']['confidence']
        impact = result['trading_impact']['total_impact_score']
        
        emoji = "📈" if action == "BUY" else "📉" if action == "SELL" else "➡️"
        
        print(f"{emoji} {symbol:4} | {action:4} | Impact: {impact:+.3f} | Confidence: {confidence:.2f}")
    
    print()
    print("SENTIMENT DISTRIBUTION:")
    print("-" * 25)
    
    sentiments = [r['sentiment']['composite_sentiment'] for r in analysis_results]
    avg_sentiment = sum(sentiments) / len(sentiments)
    positive_count = sum(1 for s in sentiments if s > 0.1)
    negative_count = sum(1 for s in sentiments if s < -0.1)
    neutral_count = len(sentiments) - positive_count - negative_count
    
    print(f"Average Market Sentiment: {avg_sentiment:.3f}")
    print(f"Positive Articles: {positive_count}/4 ({positive_count/4*100:.0f}%)")
    print(f"Negative Articles: {negative_count}/4 ({negative_count/4*100:.0f}%)")
    print(f"Neutral Articles: {neutral_count}/4 ({neutral_count/4*100:.0f}%)")
    print()
    
    print("KEY INSIGHTS:")
    print("-" * 15)
    
    # Find most impactful article
    most_impactful = max(analysis_results, key=lambda x: abs(x['trading_impact']['total_impact_score']))
    print(f"• Most Impactful: {most_impactful['article']['symbol']} with impact score {most_impactful['trading_impact']['total_impact_score']:+.3f}")
    
    # Count signal types
    all_earnings_signals = []
    all_rating_changes = []
    for result in analysis_results:
        all_earnings_signals.extend(result['signals']['earnings_signals'])
        all_rating_changes.extend(result['signals']['rating_changes'])
    
    print(f"• Earnings Signals Detected: {len(all_earnings_signals)}")
    print(f"• Rating Changes Detected: {len(all_rating_changes)}")
    
    # Market regime assessment
    if avg_sentiment > 0.2:
        regime = "BULLISH"
    elif avg_sentiment < -0.2:
        regime = "BEARISH"
    else:
        regime = "MIXED/NEUTRAL"
    
    print(f"• Overall Market Regime: {regime}")
    print()
    
    print("SYSTEM PERFORMANCE:")
    print("-" * 20)
    print(f"• Articles Processed: {len(SAMPLE_NEWS_ARTICLES)}")
    print(f"• Entities Extracted: {sum(len(r['entities']) for r in analysis_results)}")
    print(f"• Market Signals Found: {sum(len([s for signals in r['signals'].values() for s in signals]) for r in analysis_results)}")
    print(f"• Average Processing Time: ~0.8 seconds per article")
    print(f"• Average Confidence: {sum(r['trading_impact']['confidence'] for r in analysis_results) / len(analysis_results):.2f}")
    print()
    
    # Save results
    with open("nlp_analysis_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "analysis_results": analysis_results,
            "summary": {
                "average_sentiment": avg_sentiment,
                "sentiment_distribution": {
                    "positive": positive_count,
                    "negative": negative_count,
                    "neutral": neutral_count
                },
                "market_regime": regime,
                "total_articles": len(analysis_results),
                "total_entities": sum(len(r['entities']) for r in analysis_results),
                "total_signals": sum(len([s for signals in r['signals'].values() for s in signals]) for r in analysis_results)
            }
        }, f, indent=2, default=str)
    
    print("✅ NLP analysis complete! Results saved to nlp_analysis_results.json")
    print()
    print("NEXT STEPS:")
    print("• Integrate sentiment scores with trading algorithms")
    print("• Backtest NLP-driven strategies")
    print("• Set up real-time news feed processing")
    print("• Configure trading signal thresholds")
    print()

if __name__ == "__main__":
    print("Starting Hive Trade NLP system demonstration...")
    print()
    asyncio.run(demonstrate_nlp_capabilities())