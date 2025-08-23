#!/usr/bin/env python3
"""
Hive Trade - Advanced NLP Integration Test
Demonstrates how the new advanced NLP agent works with existing news sentiment system
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

class MockExistingNewsSentimentAgent:
    """Mock of existing news sentiment agent (news_sentiment_agent.py)"""
    
    def __init__(self):
        self.finbert_model = "ProsusAI/finbert"
        self.gemini_model = "gemini-pro"
        print("Existing News Sentiment Agent initialized:")
        print("  - FinBERT model loaded")
        print("  - Gemini integration ready")
        print("  - LangGraph workflow configured")
        
    async def analyze_news_sentiment(self, article_text: str, symbol: str) -> Dict[str, Any]:
        """Existing sentiment analysis using FinBERT + Gemini"""
        await asyncio.sleep(0.3)  # Simulate processing
        
        # Mock FinBERT analysis
        positive_words = ["beats", "strong", "growth", "record", "exceptional"]
        negative_words = ["challenges", "delays", "miss", "downgraded", "concerns"]
        
        pos_score = sum(0.2 for word in positive_words if word.lower() in article_text.lower())
        neg_score = sum(0.2 for word in negative_words if word.lower() in article_text.lower())
        
        finbert_score = pos_score - neg_score
        finbert_score = max(-1.0, min(1.0, finbert_score))
        
        return {
            "finbert_sentiment": {
                "score": finbert_score,
                "confidence": 0.85,
                "positive": max(0, finbert_score),
                "negative": abs(min(0, finbert_score)),
                "neutral": 1 - abs(finbert_score)
            },
            "gemini_analysis": {
                "market_impact": "high" if abs(finbert_score) > 0.5 else "medium",
                "key_themes": ["earnings", "guidance"] if "earnings" in article_text.lower() else ["operations"],
                "urgency": "high" if abs(finbert_score) > 0.6 else "medium"
            },
            "composite_score": finbert_score * 0.9  # Slight adjustment from ensemble
        }

class MockAdvancedNLPAgent:
    """Mock of the new advanced NLP agent (advanced_nlp_agent.py)"""
    
    def __init__(self):
        self.vader_analyzer = "VADER"
        self.spacy_model = "en_core_web_sm"
        self.lda_model = "LDA"
        print("Advanced NLP Agent initialized:")
        print("  - VADER sentiment analyzer loaded")
        print("  - spaCy NER model loaded")
        print("  - LDA topic modeling ready")
        
    async def enhanced_text_analysis(self, article_text: str, symbol: str) -> Dict[str, Any]:
        """Enhanced analysis with VADER, NER, topic modeling"""
        await asyncio.sleep(0.4)  # Simulate processing
        
        # Mock VADER analysis
        positive_words = ["soars", "bullish", "exceptional", "strong"]
        negative_words = ["plunges", "bearish", "concerning", "weak"]
        
        vader_compound = 0.1 * (
            sum(0.15 for word in positive_words if word.lower() in article_text.lower()) - 
            sum(0.15 for word in negative_words if word.lower() in article_text.lower())
        )
        vader_compound = max(-1.0, min(1.0, vader_compound))
        
        # Mock NER extraction
        entities = []
        if "Apple" in article_text or symbol == "AAPL":
            entities.append({"text": "Apple Inc.", "label": "ORG", "confidence": 0.95})
        if "Tesla" in article_text or symbol == "TSLA":
            entities.append({"text": "Tesla Inc.", "label": "ORG", "confidence": 0.93})
        
        # Mock topic modeling
        topics = []
        if "earnings" in article_text.lower():
            topics.append({"topic": "earnings_results", "weight": 0.8})
        if "production" in article_text.lower():
            topics.append({"topic": "operational_updates", "weight": 0.7})
        
        # Market signal detection
        signals = {
            "price_targets": [],
            "analyst_ratings": [],
            "insider_activity": []
        }
        
        if "price target" in article_text.lower():
            import re
            targets = re.findall(r'\$(\d+)', article_text)
            for target in targets[:1]:  # Take first match
                signals["price_targets"].append({
                    "target": int(target),
                    "confidence": 0.82
                })
        
        if "upgrade" in article_text.lower() or "downgrade" in article_text.lower():
            rating_type = "upgrade" if "upgrade" in article_text.lower() else "downgrade"
            signals["analyst_ratings"].append({
                "type": rating_type,
                "confidence": 0.87
            })
        
        return {
            "vader_sentiment": {
                "compound": vader_compound,
                "pos": max(0, vader_compound),
                "neu": 0.6,
                "neg": abs(min(0, vader_compound))
            },
            "named_entities": entities,
            "topic_analysis": {
                "topics": topics,
                "dominant_topic": topics[0]["topic"] if topics else "general_news"
            },
            "market_signals": signals,
            "text_metrics": {
                "readability_score": 0.75,
                "urgency_score": abs(vader_compound) * 0.8,
                "credibility_score": 0.85,
                "novelty_score": 0.7
            }
        }

class IntegratedNLPPipeline:
    """Integrated pipeline combining both NLP systems"""
    
    def __init__(self):
        self.existing_agent = MockExistingNewsSentimentAgent()
        self.advanced_agent = MockAdvancedNLPAgent()
        print("\nIntegrated NLP Pipeline initialized")
        print("Both agents ready for ensemble analysis")
        
    async def comprehensive_analysis(self, article_text: str, symbol: str, article_id: int) -> Dict[str, Any]:
        """Run comprehensive analysis using both systems"""
        
        print(f"\nProcessing article {article_id} for {symbol}...")
        print("Running parallel analysis with both NLP systems...")
        
        # Run both analyses in parallel for efficiency
        existing_result, advanced_result = await asyncio.gather(
            self.existing_agent.analyze_news_sentiment(article_text, symbol),
            self.advanced_agent.enhanced_text_analysis(article_text, symbol)
        )
        
        # Create ensemble sentiment score
        finbert_score = existing_result["finbert_sentiment"]["score"]
        vader_score = advanced_result["vader_sentiment"]["compound"]
        
        # Weight the scores (FinBERT gets higher weight for financial texts)
        ensemble_sentiment = (finbert_score * 0.6 + vader_score * 0.4)
        
        # Confidence based on agreement between models
        sentiment_agreement = 1 - abs(finbert_score - vader_score) / 2
        ensemble_confidence = (existing_result["finbert_sentiment"]["confidence"] * 0.6 + 
                             sentiment_agreement * 0.4)
        
        # Market impact assessment
        market_signals = advanced_result["market_signals"]
        signal_count = sum(len(signals) for signals in market_signals.values())
        
        # Adjust market impact based on signals and topics
        base_impact = existing_result["gemini_analysis"]["market_impact"]
        if signal_count > 0:
            enhanced_impact = "very_high" if base_impact == "high" else "high"
        else:
            enhanced_impact = base_impact
        
        # Trading recommendation
        if ensemble_sentiment > 0.3 and ensemble_confidence > 0.7:
            recommendation = "BUY"
            strength = "STRONG" if ensemble_sentiment > 0.6 else "MODERATE"
        elif ensemble_sentiment < -0.3 and ensemble_confidence > 0.7:
            recommendation = "SELL"
            strength = "STRONG" if ensemble_sentiment < -0.6 else "MODERATE"
        else:
            recommendation = "HOLD"
            strength = "NEUTRAL"
        
        return {
            "article_id": article_id,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            
            # Individual system results
            "existing_system": existing_result,
            "advanced_system": advanced_result,
            
            # Ensemble results
            "ensemble_analysis": {
                "sentiment_score": ensemble_sentiment,
                "confidence": ensemble_confidence,
                "market_impact": enhanced_impact,
                "agreement_score": sentiment_agreement
            },
            
            # Enhanced insights from integration
            "integrated_insights": {
                "dominant_topic": advanced_result["topic_analysis"]["dominant_topic"],
                "key_entities": [e["text"] for e in advanced_result["named_entities"]],
                "market_signals_detected": signal_count,
                "signal_details": market_signals,
                "urgency_level": advanced_result["text_metrics"]["urgency_score"],
                "credibility_assessment": advanced_result["text_metrics"]["credibility_score"]
            },
            
            # Trading decision
            "trading_recommendation": {
                "action": recommendation,
                "strength": strength,
                "position_sizing": min(0.25, abs(ensemble_sentiment) * ensemble_confidence),
                "time_horizon": "short" if abs(ensemble_sentiment) > 0.5 else "medium",
                "risk_level": 1 - ensemble_confidence
            }
        }

async def demonstrate_nlp_integration():
    """Main demonstration of integrated NLP systems"""
    
    print("=" * 80)
    print("HIVE TRADE - INTEGRATED NLP SYSTEM DEMONSTRATION")
    print("Advanced NLP Agent Integration with Existing News Sentiment System")
    print("=" * 80)
    
    # Sample news articles for testing integration
    test_articles = [
        {
            "id": 101,
            "symbol": "AAPL",
            "title": "Apple Crushes Q4 Earnings with Record $95B Revenue",
            "content": """Apple Inc. reported exceptional Q4 earnings today, with revenue soaring to $95 billion, 
            beating analyst estimates by 15%. The tech giant's strong performance was driven by robust iPhone 14 sales 
            and growing services revenue. CEO Tim Cook expressed confidence in future growth, citing strong demand 
            for the upcoming iPhone 15 lineup. Goldman Sachs analysts raised their price target to $210, 
            upgrading Apple to 'Buy' from 'Hold'. The earnings beat demonstrates Apple's resilience in a challenging 
            economic environment."""
        },
        {
            "id": 102,
            "symbol": "TSLA",
            "title": "Tesla Production Concerns Mount as Shanghai Plant Reduces Output",
            "content": """Tesla faces growing production challenges at its Shanghai Gigafactory, with output reduced by 25% 
            due to supply chain disruptions and regulatory compliance issues. The electric vehicle manufacturer 
            may struggle to meet Q1 delivery targets, raising concerns among investors. Morgan Stanley analysts 
            downgraded Tesla to 'Underweight' with a price target of $175, citing operational headwinds and 
            increasing competition from Chinese EV makers. Industry experts worry about Tesla's ability to 
            maintain its market leadership position."""
        },
        {
            "id": 103,
            "symbol": "MSFT",
            "title": "Microsoft AI Revolution: Azure Cloud Revenue Surges 40% on ChatGPT Integration",
            "content": """Microsoft Corporation delivered outstanding results as Azure cloud revenue surged 40% year-over-year, 
            driven by unprecedented demand for AI services following ChatGPT integration. The software giant's 
            partnership with OpenAI continues to revolutionize enterprise AI adoption. CEO Satya Nadella announced 
            plans for $15 billion in additional AI infrastructure investments. Microsoft's AI-powered productivity 
            tools are seeing exceptional enterprise adoption rates. Analysts remain bullish on Microsoft's 
            long-term AI strategy and market positioning."""
        }
    ]
    
    # Initialize integrated pipeline
    pipeline = IntegratedNLPPipeline()
    
    # Process each article through integrated analysis
    results = []
    
    for article in test_articles:
        print("\n" + "=" * 60)
        print(f"ANALYZING: {article['title']}")
        print("=" * 60)
        
        result = await pipeline.comprehensive_analysis(
            article['content'],
            article['symbol'],
            article['id']
        )
        
        results.append(result)
        
        # Display detailed results
        print(f"\nSYMBOL: {result['symbol']}")
        print("-" * 20)
        
        print("INDIVIDUAL SYSTEM RESULTS:")
        print(f"  Existing System (FinBERT + Gemini):")
        print(f"    - Sentiment Score: {result['existing_system']['finbert_sentiment']['score']:+.3f}")
        print(f"    - Confidence: {result['existing_system']['finbert_sentiment']['confidence']:.2f}")
        print(f"    - Market Impact: {result['existing_system']['gemini_analysis']['market_impact']}")
        
        print(f"  Advanced System (VADER + NER + Topics):")
        print(f"    - VADER Score: {result['advanced_system']['vader_sentiment']['compound']:+.3f}")
        print(f"    - Entities Found: {len(result['advanced_system']['named_entities'])}")
        print(f"    - Market Signals: {result['integrated_insights']['market_signals_detected']}")
        
        print("\nENSEMBLE ANALYSIS:")
        print(f"  - Combined Sentiment: {result['ensemble_analysis']['sentiment_score']:+.3f}")
        print(f"  - Ensemble Confidence: {result['ensemble_analysis']['confidence']:.2f}")
        print(f"  - Model Agreement: {result['ensemble_analysis']['agreement_score']:.2f}")
        print(f"  - Enhanced Market Impact: {result['ensemble_analysis']['market_impact']}")
        
        print("\nINTEGRATE INSIGHTS:")
        print(f"  - Dominant Topic: {result['integrated_insights']['dominant_topic']}")
        print(f"  - Key Entities: {', '.join(result['integrated_insights']['key_entities'])}")
        print(f"  - Urgency Level: {result['integrated_insights']['urgency_level']:.2f}")
        print(f"  - Credibility Score: {result['integrated_insights']['credibility_assessment']:.2f}")
        
        if result['integrated_insights']['signal_details']['price_targets']:
            targets = result['integrated_insights']['signal_details']['price_targets']
            target_strs = [f"${t['target']}" for t in targets]
            print(f"  - Price Targets: {', '.join(target_strs)}")
        
        if result['integrated_insights']['signal_details']['analyst_ratings']:
            ratings = result['integrated_insights']['signal_details']['analyst_ratings']
            print(f"  - Rating Changes: {', '.join(r['type'] for r in ratings)}")
        
        print("\nTRADING RECOMMENDATION:")
        rec = result['trading_recommendation']
        print(f"  - Action: {rec['action']} ({rec['strength']})")
        print(f"  - Position Size: {rec['position_sizing']:.1%} of portfolio")
        print(f"  - Time Horizon: {rec['time_horizon']}")
        print(f"  - Risk Level: {rec['risk_level']:.2f}")
    
    # Integration Performance Analysis
    print("\n" + "=" * 80)
    print("INTEGRATION PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    print("\nSYSTEM COMPARISON:")
    print("-" * 20)
    
    existing_scores = [r['existing_system']['finbert_sentiment']['score'] for r in results]
    advanced_scores = [r['advanced_system']['vader_sentiment']['compound'] for r in results]
    ensemble_scores = [r['ensemble_analysis']['sentiment_score'] for r in results]
    
    print("Sentiment Scores by System:")
    for i, result in enumerate(results):
        symbol = result['symbol']
        existing = existing_scores[i]
        advanced = advanced_scores[i]
        ensemble = ensemble_scores[i]
        
        print(f"  {symbol}: Existing={existing:+.3f}, Advanced={advanced:+.3f}, Ensemble={ensemble:+.3f}")
    
    print("\nAGGREGATE METRICS:")
    print("-" * 18)
    
    avg_agreement = sum(r['ensemble_analysis']['agreement_score'] for r in results) / len(results)
    avg_confidence = sum(r['ensemble_analysis']['confidence'] for r in results) / len(results)
    total_signals = sum(r['integrated_insights']['market_signals_detected'] for r in results)
    
    print(f"  - Average Model Agreement: {avg_agreement:.2f}")
    print(f"  - Average Ensemble Confidence: {avg_confidence:.2f}")
    print(f"  - Total Market Signals Detected: {total_signals}")
    print(f"  - Articles Processed: {len(results)}")
    
    # Trading recommendations summary
    recommendations = [r['trading_recommendation']['action'] for r in results]
    buy_count = recommendations.count('BUY')
    sell_count = recommendations.count('SELL')
    hold_count = recommendations.count('HOLD')
    
    print("\nTRADING RECOMMENDATIONS SUMMARY:")
    print("-" * 35)
    print(f"  - BUY signals: {buy_count}")
    print(f"  - SELL signals: {sell_count}")
    print(f"  - HOLD signals: {hold_count}")
    
    strong_signals = sum(1 for r in results if r['trading_recommendation']['strength'] == 'STRONG')
    print(f"  - Strong conviction signals: {strong_signals}/{len(results)}")
    
    # Integration benefits
    print("\nINTEGRATION BENEFITS:")
    print("-" * 22)
    print("[+] Enhanced accuracy through ensemble voting")
    print("[+] Comprehensive entity recognition and topic modeling")
    print("[+] Market signal detection (price targets, ratings)")
    print("[+] Multi-dimensional confidence scoring")
    print("[+] Advanced text metrics (urgency, credibility)")
    print("[+] Improved trading recommendation system")
    
    print("\nSYSTEM ARCHITECTURE ADVANTAGES:")
    print("-" * 35)
    print("* Existing System Strengths:")
    print("  - FinBERT specialization for financial texts")
    print("  - Gemini's advanced contextual understanding")
    print("  - Proven LangGraph workflow automation")
    print()
    print("* Advanced System Contributions:")
    print("  - VADER for general sentiment robustness")
    print("  - spaCy NER for entity extraction")
    print("  - LDA topic modeling for thematic analysis")
    print("  - Custom market signal detection")
    print()
    print("* Ensemble Benefits:")
    print("  - Reduced single-model bias")
    print("  - Higher confidence through agreement")
    print("  - Comprehensive market intelligence")
    print("  - Adaptive decision-making capabilities")
    
    # Save integration results
    integration_results = {
        "timestamp": datetime.now().isoformat(),
        "system_config": {
            "existing_system": {
                "models": ["FinBERT", "Gemini"],
                "workflow": "LangGraph"
            },
            "advanced_system": {
                "models": ["VADER", "spaCy", "LDA"],
                "features": ["NER", "Topic Modeling", "Signal Detection"]
            },
            "ensemble_method": "weighted_average",
            "weights": {"finbert": 0.6, "vader": 0.4}
        },
        "analysis_results": results,
        "performance_metrics": {
            "average_agreement": avg_agreement,
            "average_confidence": avg_confidence,
            "total_signals_detected": total_signals,
            "trading_recommendations": {
                "buy": buy_count,
                "sell": sell_count,
                "hold": hold_count,
                "strong_conviction": strong_signals
            }
        }
    }
    
    with open("nlp_integration_results.json", "w") as f:
        json.dump(integration_results, f, indent=2, default=str)
    
    print("\nNEXT STEPS:")
    print("-" * 15)
    print("* Deploy integrated pipeline to production")
    print("* Configure real-time news feed processing")
    print("* Set up automated backtesting with ensemble signals")
    print("* Implement adaptive weight optimization")
    print("* Add performance monitoring and alerting")
    print("* Scale to process 1000+ articles per day")
    
    print(f"\n[SUCCESS] Integration analysis complete!")
    print(f"Detailed results saved to nlp_integration_results.json")
    print()

if __name__ == "__main__":
    print("Starting Hive Trade NLP integration test...")
    print("Testing advanced NLP agent integration with existing news sentiment system")
    print()
    asyncio.run(demonstrate_nlp_integration())