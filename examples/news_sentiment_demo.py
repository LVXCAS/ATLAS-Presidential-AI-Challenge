#!/usr/bin/env python3
"""
News and Sentiment Analysis Agent - Demo Script

This script demonstrates the capabilities of the News and Sentiment Analysis Agent
including news ingestion, sentiment analysis, and event detection.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# Add project root to path
sys.path.append('.')

from agents.news_sentiment_agent import (
    create_news_sentiment_agent,
    NewsSource,
    EventType
)
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsSentimentDemo:
    """Demo class for News and Sentiment Analysis Agent"""
    
    def __init__(self):
        self.agent = None
    
    async def setup(self):
        """Initialize the agent"""
        logger.info("🚀 Initializing News and Sentiment Analysis Agent...")
        
        try:
            self.agent = await create_news_sentiment_agent()
            logger.info("✅ Agent initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent: {e}")
            return False
    
    def print_separator(self, title: str):
        """Print a formatted separator"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
    
    def print_sentiment_analysis(self, analyses: List[Dict[str, Any]]):
        """Print sentiment analysis results in a formatted way"""
        if not analyses:
            print("No sentiment analyses found.")
            return
        
        print(f"\n📊 SENTIMENT ANALYSIS RESULTS ({len(analyses)} analyses)")
        print("-" * 60)
        
        # Group by symbol
        by_symbol = {}
        for analysis in analyses:
            symbol = analysis['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(analysis)
        
        for symbol, symbol_analyses in by_symbol.items():
            print(f"\n🏢 {symbol}:")
            
            # Calculate average sentiment
            avg_composite = sum(a['composite_score'] for a in symbol_analyses) / len(symbol_analyses)
            avg_confidence = sum(a['confidence_level'] for a in symbol_analyses) / len(symbol_analyses)
            
            sentiment_label = "POSITIVE" if avg_composite > 0.5 else "NEGATIVE" if avg_composite < -0.5 else "NEUTRAL"
            sentiment_emoji = "📈" if avg_composite > 0.5 else "📉" if avg_composite < -0.5 else "➡️"
            
            print(f"   Overall Sentiment: {sentiment_emoji} {sentiment_label} ({avg_composite:.2f})")
            print(f"   Average Confidence: {avg_confidence:.1%}")
            print(f"   Articles Analyzed: {len(symbol_analyses)}")
            
            # Show top sentiment analysis
            if symbol_analyses:
                top_analysis = max(symbol_analyses, key=lambda x: abs(x['composite_score']))
                print(f"   Strongest Signal: {top_analysis['composite_score']:.2f} ({top_analysis['composite_label']})")
                
                if top_analysis.get('gemini_reasoning'):
                    reasoning = top_analysis['gemini_reasoning'][:100] + "..." if len(top_analysis['gemini_reasoning']) > 100 else top_analysis['gemini_reasoning']
                    print(f"   AI Reasoning: {reasoning}")
    
    def print_detected_events(self, events: List[Dict[str, Any]]):
        """Print detected events in a formatted way"""
        if not events:
            print("No market events detected.")
            return
        
        print(f"\n🚨 DETECTED MARKET EVENTS ({len(events)} events)")
        print("-" * 60)
        
        for i, event in enumerate(events, 1):
            event_emoji = {
                'earnings': '💰',
                'merger_acquisition': '🤝',
                'regulatory': '⚖️',
                'economic_data': '📊',
                'company_news': '📰',
                'market_moving': '🌊'
            }.get(event['event_type'], '📢')
            
            direction_emoji = {
                'bullish': '🐂',
                'bearish': '🐻',
                'neutral': '➡️'
            }.get(event['predicted_direction'], '❓')
            
            print(f"\n{i}. {event_emoji} {event['title']}")
            print(f"   Type: {event['event_type'].replace('_', ' ').title()}")
            print(f"   Symbols: {', '.join(event['symbols'])}")
            print(f"   Impact Score: {event['impact_score']:.2f}/1.0")
            print(f"   Direction: {direction_emoji} {event['predicted_direction'].title()}")
            print(f"   Time Horizon: {event['time_horizon'].replace('_', ' ').title()}")
            print(f"   Confidence: {event['confidence']:.1%}")
            
            if event.get('description'):
                desc = event['description'][:150] + "..." if len(event['description']) > 150 else event['description']
                print(f"   Description: {desc}")
    
    def print_statistics(self, stats: Dict[str, Any]):
        """Print processing statistics"""
        print(f"\n📈 PROCESSING STATISTICS")
        print("-" * 40)
        
        print(f"Symbols Requested: {stats.get('symbols_requested', 0)}")
        print(f"Sources Used: {stats.get('sources_requested', 0)}")
        print(f"Articles Fetched: {stats.get('articles_fetched', 0)}")
        print(f"Articles Processed: {stats.get('articles_processed', 0)}")
        print(f"Sentiment Analyses: {stats.get('sentiment_analyses', 0)}")
        print(f"Events Detected: {stats.get('events_detected', 0)}")
        print(f"Processing Time: {stats.get('duration', 0):.2f} seconds")
        
        if stats.get('sentiment_distribution'):
            dist = stats['sentiment_distribution']
            print(f"\nSentiment Distribution:")
            print(f"  📈 Positive: {dist.get('positive', 0)}")
            print(f"  ➡️ Neutral: {dist.get('neutral', 0)}")
            print(f"  📉 Negative: {dist.get('negative', 0)}")
        
        if stats.get('average_confidence'):
            print(f"Average Confidence: {stats['average_confidence']:.1%}")
    
    async def demo_basic_analysis(self):
        """Demonstrate basic sentiment analysis"""
        self.print_separator("BASIC SENTIMENT ANALYSIS DEMO")
        
        print("Analyzing sentiment for major tech stocks...")
        print("Symbols: AAPL, GOOGL, MSFT, TSLA, AMZN")
        print("Time Range: Last 12 hours")
        print("Sources: RSS Feeds, Yahoo Finance")
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        try:
            result = await self.agent.analyze_news_sentiment(
                symbols=symbols,
                hours_back=12,
                sources=[NewsSource.RSS_FEED, NewsSource.YAHOO_FINANCE]
            )
            
            if result['success']:
                print("✅ Analysis completed successfully!")
                
                self.print_statistics(result['statistics'])
                self.print_sentiment_analysis(result['sentiment_analyses'])
                self.print_detected_events(result['detected_events'])
                
                return result
            else:
                print(f"❌ Analysis failed: {result.get('errors', [])}")
                return None
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            return None
    
    async def demo_comprehensive_analysis(self):
        """Demonstrate comprehensive analysis with more symbols and sources"""
        self.print_separator("COMPREHENSIVE MARKET SENTIMENT ANALYSIS")
        
        print("Analyzing sentiment across multiple sectors...")
        
        symbols = [
            # Tech
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB',
            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX'
        ]
        
        print(f"Symbols: {len(symbols)} stocks across multiple sectors")
        print("Time Range: Last 24 hours")
        print("Sources: RSS Feeds, Yahoo Finance, MarketWatch")
        
        try:
            result = await self.agent.analyze_news_sentiment(
                symbols=symbols,
                hours_back=24,
                sources=[NewsSource.RSS_FEED, NewsSource.YAHOO_FINANCE, NewsSource.MARKETWATCH]
            )
            
            if result['success']:
                print("✅ Comprehensive analysis completed!")
                
                self.print_statistics(result['statistics'])
                
                # Show top positive and negative sentiments
                analyses = result['sentiment_analyses']
                if analyses:
                    # Sort by composite score
                    sorted_analyses = sorted(analyses, key=lambda x: x['composite_score'], reverse=True)
                    
                    print(f"\n🏆 TOP 5 MOST POSITIVE SENTIMENTS")
                    print("-" * 50)
                    for i, analysis in enumerate(sorted_analyses[:5], 1):
                        print(f"{i}. {analysis['symbol']}: {analysis['composite_score']:.2f} ({analysis['composite_label']})")
                    
                    print(f"\n⚠️ TOP 5 MOST NEGATIVE SENTIMENTS")
                    print("-" * 50)
                    for i, analysis in enumerate(sorted_analyses[-5:], 1):
                        print(f"{i}. {analysis['symbol']}: {analysis['composite_score']:.2f} ({analysis['composite_label']})")
                
                self.print_detected_events(result['detected_events'])
                
                return result
            else:
                print(f"❌ Analysis failed: {result.get('errors', [])}")
                return None
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            return None
    
    async def demo_event_detection(self):
        """Demonstrate event detection capabilities"""
        self.print_separator("MARKET EVENT DETECTION DEMO")
        
        print("Scanning for market-moving events...")
        print("Focus: High-impact events across major stocks")
        print("Time Range: Last 48 hours")
        
        # Focus on stocks that often have news
        symbols = ['AAPL', 'TSLA', 'AMZN', 'GOOGL', 'MSFT', 'META', 'NVDA', 'NFLX']
        
        try:
            result = await self.agent.analyze_news_sentiment(
                symbols=symbols,
                hours_back=48,
                sources=[NewsSource.RSS_FEED, NewsSource.YAHOO_FINANCE, NewsSource.MARKETWATCH]
            )
            
            if result['success']:
                print("✅ Event detection completed!")
                
                events = result['detected_events']
                
                if events:
                    # Group events by type
                    by_type = {}
                    for event in events:
                        event_type = event['event_type']
                        if event_type not in by_type:
                            by_type[event_type] = []
                        by_type[event_type].append(event)
                    
                    print(f"\n📊 EVENT SUMMARY BY TYPE")
                    print("-" * 40)
                    for event_type, type_events in by_type.items():
                        print(f"{event_type.replace('_', ' ').title()}: {len(type_events)} events")
                    
                    # Show high-impact events
                    high_impact_events = [e for e in events if e['impact_score'] >= 0.7]
                    
                    if high_impact_events:
                        print(f"\n🚨 HIGH-IMPACT EVENTS (Impact Score ≥ 0.7)")
                        print("-" * 60)
                        
                        for event in high_impact_events:
                            print(f"• {event['title']}")
                            print(f"  Symbols: {', '.join(event['symbols'])}")
                            print(f"  Impact: {event['impact_score']:.2f}, Direction: {event['predicted_direction']}")
                            print()
                    
                    self.print_detected_events(events)
                else:
                    print("No significant events detected in the specified time range.")
                
                return result
            else:
                print(f"❌ Event detection failed: {result.get('errors', [])}")
                return None
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            return None
    
    async def demo_performance_test(self):
        """Demonstrate performance with larger dataset"""
        self.print_separator("PERFORMANCE TEST - LARGE BATCH PROCESSING")
        
        print("Testing performance with large dataset...")
        print("This simulates processing towards the 1000 article requirement")
        
        # Use many symbols to increase article volume
        symbols = [
            # S&P 500 top 50 by market cap (sample)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'MA', 'CVX', 'ABBV', 'PFE',
            'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT',
            'CRM', 'ACN', 'NFLX', 'ADBE', 'VZ', 'CMCSA', 'DHR', 'NKE', 'TXN',
            'NEE', 'RTX', 'QCOM', 'PM', 'SPGI', 'HON', 'UNP', 'T', 'LOW',
            'IBM', 'GS', 'AMGN', 'CAT', 'SBUX', 'INTU', 'AMD', 'BKNG'
        ]
        
        print(f"Symbols: {len(symbols)} major stocks")
        print("Time Range: Last 72 hours")
        print("Sources: All available sources")
        
        start_time = datetime.utcnow()
        
        try:
            result = await self.agent.analyze_news_sentiment(
                symbols=symbols,
                hours_back=72,
                sources=[NewsSource.RSS_FEED, NewsSource.YAHOO_FINANCE, NewsSource.MARKETWATCH]
            )
            
            end_time = datetime.utcnow()
            total_time = (end_time - start_time).total_seconds()
            
            if result['success']:
                print("✅ Performance test completed!")
                
                stats = result['statistics']
                articles_processed = stats.get('articles_processed', 0)
                sentiment_analyses = len(result.get('sentiment_analyses', []))
                events_detected = len(result.get('detected_events', []))
                
                print(f"\n⚡ PERFORMANCE METRICS")
                print("-" * 40)
                print(f"Total Processing Time: {total_time:.2f} seconds")
                print(f"Articles Processed: {articles_processed}")
                print(f"Sentiment Analyses: {sentiment_analyses}")
                print(f"Events Detected: {events_detected}")
                
                if articles_processed > 0:
                    print(f"Processing Rate: {articles_processed/total_time:.2f} articles/second")
                    print(f"Analysis Rate: {sentiment_analyses/total_time:.2f} analyses/second")
                
                # Performance assessment
                if articles_processed >= 50:
                    print("🎉 EXCELLENT: Processed substantial number of articles")
                elif articles_processed >= 20:
                    print("✅ GOOD: Processed reasonable number of articles")
                elif articles_processed >= 10:
                    print("⚠️ MODERATE: Processed some articles")
                else:
                    print("❌ LOW: Few articles processed (may be due to limited news)")
                
                if total_time <= 60:
                    print("🚀 FAST: Completed within 1 minute")
                elif total_time <= 180:
                    print("✅ REASONABLE: Completed within 3 minutes")
                else:
                    print("⏰ SLOW: Took longer than expected")
                
                return result
            else:
                print(f"❌ Performance test failed: {result.get('errors', [])}")
                return None
                
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return None
    
    async def run_all_demos(self):
        """Run all demonstration scenarios"""
        print("🎬 NEWS AND SENTIMENT ANALYSIS AGENT - COMPREHENSIVE DEMO")
        print("=" * 80)
        print("This demo showcases the capabilities of the News and Sentiment Analysis Agent")
        print("including news ingestion, sentiment analysis, and event detection.")
        print("=" * 80)
        
        # Setup
        if not await self.setup():
            print("❌ Failed to initialize agent. Aborting demo.")
            return False
        
        demos = [
            ("Basic Analysis", self.demo_basic_analysis),
            ("Comprehensive Analysis", self.demo_comprehensive_analysis),
            ("Event Detection", self.demo_event_detection),
            ("Performance Test", self.demo_performance_test)
        ]
        
        results = {}
        
        for demo_name, demo_func in demos:
            print(f"\n🎯 Starting {demo_name}...")
            
            try:
                result = await demo_func()
                results[demo_name] = result
                
                if result:
                    print(f"✅ {demo_name} completed successfully!")
                else:
                    print(f"⚠️ {demo_name} completed with issues.")
                
            except Exception as e:
                print(f"❌ {demo_name} failed: {e}")
                results[demo_name] = None
        
        # Final summary
        self.print_separator("DEMO SUMMARY")
        
        successful_demos = sum(1 for result in results.values() if result and result.get('success'))
        total_demos = len(demos)
        
        print(f"Completed Demos: {successful_demos}/{total_demos}")
        
        for demo_name, result in results.items():
            if result and result.get('success'):
                print(f"✅ {demo_name}")
            else:
                print(f"❌ {demo_name}")
        
        if successful_demos == total_demos:
            print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
            print("The News and Sentiment Analysis Agent is working correctly.")
        else:
            print(f"\n⚠️ {total_demos - successful_demos} DEMOS HAD ISSUES")
            print("Please check the logs for details.")
        
        # Save demo results
        demo_summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'successful_demos': successful_demos,
            'total_demos': total_demos,
            'demo_results': {k: v is not None and v.get('success', False) for k, v in results.items()}
        }
        
        with open('news_sentiment_demo_results.json', 'w') as f:
            json.dump(demo_summary, f, indent=2, default=str)
        
        print(f"\n📄 Demo results saved to news_sentiment_demo_results.json")
        
        return successful_demos == total_demos

async def main():
    """Main demo execution"""
    demo = NewsSentimentDemo()
    
    try:
        success = await demo.run_all_demos()
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\n❌ Demo execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())