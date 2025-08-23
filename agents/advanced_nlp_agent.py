"""
Advanced NLP Agent - Enhanced Natural Language Processing
Includes advanced text analysis, topic modeling, entity recognition, and market signal extraction
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
import numpy as np
from collections import defaultdict, Counter

# Advanced NLP imports
try:
    import spacy
    import torch
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertModel
    )
    from sentence_transformers import SentenceTransformer
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    spacy = None
    torch = None

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    import yfinance as yf
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    EXTRA_LIBS_AVAILABLE = True
except ImportError:
    EXTRA_LIBS_AVAILABLE = False

# Topic modeling
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class EntityType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "GPE"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    DATE = "DATE"
    TIME = "TIME"
    PRODUCT = "PRODUCT"

class TopicCategory(Enum):
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    MARKET_SENTIMENT = "market_sentiment"
    ECONOMIC_INDICATORS = "economic_indicators"
    COMPANY_STRATEGY = "company_strategy"
    ANALYST_RATINGS = "analyst_ratings"
    INSIDER_TRADING = "insider_trading"

class MarketSignalStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

@dataclass
class NamedEntity:
    """Extracted named entity"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    context: str

@dataclass
class TopicAnalysis:
    """Topic modeling result"""
    topic_id: int
    category: TopicCategory
    keywords: List[str]
    weight: float
    coherence_score: float
    documents_count: int

@dataclass
class MarketSignal:
    """Detected market signal from text"""
    signal_type: str
    strength: MarketSignalStrength
    direction: str  # bullish, bearish, neutral
    confidence: float
    time_horizon: str  # immediate, short_term, long_term
    reasoning: str
    extracted_values: Dict[str, Any]
    relevant_entities: List[str]

@dataclass
class AdvancedTextAnalysis:
    """Comprehensive text analysis result"""
    article_id: int
    symbol: str
    
    # Basic metrics
    word_count: int
    sentence_count: int
    readability_score: float
    complexity_grade: float
    
    # Sentiment analysis
    vader_sentiment: Dict[str, float]
    finbert_sentiment: Dict[str, float]
    custom_sentiment: Dict[str, float]
    composite_sentiment: float
    sentiment_confidence: float
    
    # Named entities
    entities: List[NamedEntity]
    key_people: List[str]
    organizations: List[str]
    financial_values: List[str]
    
    # Topic analysis
    topics: List[TopicAnalysis]
    primary_topic: str
    topic_confidence: float
    
    # Market signals
    market_signals: List[MarketSignal]
    signal_strength: float
    trading_recommendation: str
    
    # Advanced features
    urgency_score: float
    credibility_score: float
    novelty_score: float
    market_impact_score: float
    
    timestamp: datetime

class AdvancedNLPProcessor:
    """Advanced NLP processing engine"""
    
    def __init__(self):
        self.nlp = None
        self.sentence_transformer = None
        self.finbert_model = None
        self.vader_analyzer = None
        self.lemmatizer = None
        self.stop_words = None
        self.tfidf_vectorizer = None
        self.lda_model = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all NLP models"""
        try:
            if ADVANCED_NLP_AVAILABLE:
                # Load spaCy model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded successfully")
                except IOError:
                    logger.warning("spaCy en_core_web_sm not found. Install with: python -m spacy download en_core_web_sm")
                
                # Load sentence transformer for embeddings
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer loaded")
                
                # Load FinBERT for financial sentiment
                finbert_model_name = "ProsusAI/finbert"
                self.finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
                logger.info("FinBERT model loaded")
            
            if EXTRA_LIBS_AVAILABLE:
                # Initialize VADER sentiment analyzer
                self.vader_analyzer = SentimentIntensityAnalyzer()
                logger.info("VADER sentiment analyzer initialized")
            
            if SKLEARN_AVAILABLE:
                # Initialize NLTK components
                try:
                    nltk.download('stopwords', quiet=True)
                    nltk.download('wordnet', quiet=True)
                    nltk.download('punkt', quiet=True)
                    
                    self.stop_words = set(stopwords.words('english'))
                    self.lemmatizer = WordNetLemmatizer()
                    logger.info("NLTK components initialized")
                except Exception as e:
                    logger.warning(f"NLTK initialization failed: {e}")
                
                # Initialize topic modeling components
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )
                self.lda_model = LatentDirichletAllocation(
                    n_components=10,
                    random_state=42,
                    learning_method='batch',
                    max_iter=25
                )
                logger.info("Topic modeling components initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
    
    async def process_text(self, text: str, symbol: str, article_id: int) -> AdvancedTextAnalysis:
        """Perform comprehensive text analysis"""
        
        # Basic text metrics
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Readability scores
        readability_score = 0.0
        complexity_grade = 0.0
        if EXTRA_LIBS_AVAILABLE:
            try:
                readability_score = flesch_reading_ease(text)
                complexity_grade = flesch_kincaid_grade(text)
            except Exception as e:
                logger.warning(f"Readability calculation failed: {e}")
        
        # Sentiment analysis
        vader_sentiment = await self._analyze_vader_sentiment(text)
        finbert_sentiment = await self._analyze_finbert_sentiment(text)
        custom_sentiment = await self._analyze_custom_sentiment(text, symbol)
        
        # Composite sentiment
        composite_sentiment, sentiment_confidence = self._calculate_composite_sentiment(
            vader_sentiment, finbert_sentiment, custom_sentiment
        )
        
        # Named entity recognition
        entities = await self._extract_named_entities(text)
        key_people, organizations, financial_values = self._categorize_entities(entities)
        
        # Topic analysis
        topics = await self._analyze_topics([text])
        primary_topic, topic_confidence = self._determine_primary_topic(topics)
        
        # Market signal detection
        market_signals = await self._detect_market_signals(text, symbol)
        signal_strength = self._calculate_signal_strength(market_signals)
        trading_recommendation = self._generate_trading_recommendation(market_signals, composite_sentiment)
        
        # Advanced scoring
        urgency_score = self._calculate_urgency_score(text, entities)
        credibility_score = self._calculate_credibility_score(text)
        novelty_score = await self._calculate_novelty_score(text, symbol)
        market_impact_score = self._calculate_market_impact_score(
            entities, market_signals, composite_sentiment
        )
        
        return AdvancedTextAnalysis(
            article_id=article_id,
            symbol=symbol,
            word_count=word_count,
            sentence_count=sentence_count,
            readability_score=readability_score,
            complexity_grade=complexity_grade,
            vader_sentiment=vader_sentiment,
            finbert_sentiment=finbert_sentiment,
            custom_sentiment=custom_sentiment,
            composite_sentiment=composite_sentiment,
            sentiment_confidence=sentiment_confidence,
            entities=entities,
            key_people=key_people,
            organizations=organizations,
            financial_values=financial_values,
            topics=topics,
            primary_topic=primary_topic,
            topic_confidence=topic_confidence,
            market_signals=market_signals,
            signal_strength=signal_strength,
            trading_recommendation=trading_recommendation,
            urgency_score=urgency_score,
            credibility_score=credibility_score,
            novelty_score=novelty_score,
            market_impact_score=market_impact_score,
            timestamp=datetime.utcnow()
        )
    
    async def _analyze_vader_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        if not self.vader_analyzer:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu'],
                "compound": scores['compound']
            }
        except Exception as e:
            logger.error(f"VADER sentiment analysis failed: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    async def _analyze_finbert_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using FinBERT"""
        if not self.finbert_model or not self.finbert_tokenizer:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        try:
            # Tokenize and predict
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT typically has labels: [negative, neutral, positive]
            return {
                "negative": float(predictions[0][0]),
                "neutral": float(predictions[0][1]),
                "positive": float(predictions[0][2])
            }
            
        except Exception as e:
            logger.error(f"FinBERT sentiment analysis failed: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    
    async def _analyze_custom_sentiment(self, text: str, symbol: str) -> Dict[str, float]:
        """Custom sentiment analysis with financial domain knowledge"""
        
        # Financial sentiment lexicon
        bullish_terms = [
            'growth', 'profit', 'revenue', 'beat', 'exceed', 'strong', 'positive',
            'upgrade', 'bullish', 'buy', 'gain', 'rise', 'increase', 'expansion',
            'outperform', 'momentum', 'breakout', 'rally', 'surge'
        ]
        
        bearish_terms = [
            'loss', 'decline', 'weak', 'negative', 'downgrade', 'bearish', 'sell',
            'fall', 'drop', 'decrease', 'contraction', 'underperform', 'crash',
            'correction', 'selloff', 'miss', 'disappoint', 'warning', 'concern'
        ]
        
        uncertainty_terms = [
            'volatile', 'uncertain', 'unclear', 'pending', 'awaiting', 'mixed',
            'cautious', 'watch', 'monitor', 'evaluate', 'consider', 'potential'
        ]
        
        text_lower = text.lower()
        
        # Count terms
        bullish_count = sum(1 for term in bullish_terms if term in text_lower)
        bearish_count = sum(1 for term in bearish_terms if term in text_lower)
        uncertainty_count = sum(1 for term in uncertainty_terms if term in text_lower)
        
        total_count = bullish_count + bearish_count + uncertainty_count
        
        if total_count == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        positive_score = bullish_count / total_count
        negative_score = bearish_count / total_count
        neutral_score = uncertainty_count / total_count
        
        # Normalize to ensure sum = 1
        total_score = positive_score + negative_score + neutral_score
        if total_score > 0:
            positive_score /= total_score
            negative_score /= total_score
            neutral_score /= total_score
        
        return {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": neutral_score
        }
    
    def _calculate_composite_sentiment(self, vader: Dict, finbert: Dict, custom: Dict) -> Tuple[float, float]:
        """Calculate weighted composite sentiment"""
        
        # Weights for different models
        vader_weight = 0.2
        finbert_weight = 0.5  # Higher weight for domain-specific model
        custom_weight = 0.3
        
        # Convert to numerical scores (-1 to 1)
        vader_score = vader.get('compound', 0.0)
        finbert_score = (finbert.get('positive', 0) - finbert.get('negative', 0))
        custom_score = (custom.get('positive', 0) - custom.get('negative', 0))
        
        # Calculate weighted average
        composite_score = (
            vader_score * vader_weight +
            finbert_score * finbert_weight +
            custom_score * custom_weight
        )
        
        # Calculate confidence based on agreement
        scores = [vader_score, finbert_score, custom_score]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        confidence = max(0.0, 1.0 - (std_score / 2.0))  # Higher agreement = higher confidence
        
        return composite_score, confidence
    
    async def _extract_named_entities(self, text: str) -> List[NamedEntity]:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Get context around entity
                start_context = max(0, ent.start - 5)
                end_context = min(len(doc), ent.end + 5)
                context = doc[start_context:end_context].text
                
                entity = NamedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=getattr(ent, 'confidence', 0.8),  # Default confidence
                    context=context
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Named entity recognition failed: {e}")
            return []
    
    def _categorize_entities(self, entities: List[NamedEntity]) -> Tuple[List[str], List[str], List[str]]:
        """Categorize entities into people, organizations, and financial values"""
        
        key_people = []
        organizations = []
        financial_values = []
        
        for entity in entities:
            if entity.label in ['PERSON']:
                key_people.append(entity.text)
            elif entity.label in ['ORG', 'ORGANIZATION']:
                organizations.append(entity.text)
            elif entity.label in ['MONEY', 'PERCENT', 'QUANTITY']:
                financial_values.append(entity.text)
        
        # Remove duplicates and sort
        key_people = sorted(list(set(key_people)))
        organizations = sorted(list(set(organizations)))
        financial_values = sorted(list(set(financial_values)))
        
        return key_people, organizations, financial_values
    
    async def _analyze_topics(self, texts: List[str]) -> List[TopicAnalysis]:
        """Perform topic modeling on texts"""
        if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer or not self.lda_model:
            return []
        
        try:
            # Preprocess texts
            processed_texts = []
            for text in texts:
                # Simple preprocessing
                text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
                if self.stop_words:
                    words = [word for word in text.split() if word not in self.stop_words]
                    text = ' '.join(words)
                processed_texts.append(text)
            
            if not processed_texts or all(not text.strip() for text in processed_texts):
                return []
            
            # Vectorize texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
            
            # Perform LDA topic modeling
            lda_output = self.lda_model.fit_transform(tfidf_matrix)
            
            # Extract topics
            topics = []
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            for topic_idx, topic in enumerate(self.lda_model.components_):
                # Get top words for this topic
                top_words_idx = topic.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_words_idx]
                
                # Classify topic category based on keywords
                category = self._classify_topic_category(keywords)
                
                # Calculate topic weight (average document probability)
                weight = float(np.mean(lda_output[:, topic_idx])) if len(lda_output) > 0 else 0.0
                
                topic_analysis = TopicAnalysis(
                    topic_id=topic_idx,
                    category=category,
                    keywords=keywords,
                    weight=weight,
                    coherence_score=0.5,  # Simplified coherence score
                    documents_count=len(texts)
                )
                topics.append(topic_analysis)
            
            return sorted(topics, key=lambda x: x.weight, reverse=True)
            
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            return []
    
    def _classify_topic_category(self, keywords: List[str]) -> TopicCategory:
        """Classify topic based on keywords"""
        
        category_keywords = {
            TopicCategory.EARNINGS: ['earnings', 'revenue', 'profit', 'quarterly', 'results'],
            TopicCategory.MERGER_ACQUISITION: ['merger', 'acquisition', 'deal', 'buyout', 'takeover'],
            TopicCategory.REGULATORY: ['regulation', 'fda', 'approval', 'compliance', 'investigation'],
            TopicCategory.MARKET_SENTIMENT: ['sentiment', 'bullish', 'bearish', 'optimistic', 'pessimistic'],
            TopicCategory.ECONOMIC_INDICATORS: ['gdp', 'inflation', 'unemployment', 'fed', 'economy'],
            TopicCategory.ANALYST_RATINGS: ['rating', 'upgrade', 'downgrade', 'analyst', 'recommendation'],
            TopicCategory.COMPANY_STRATEGY: ['strategy', 'expansion', 'growth', 'investment', 'initiative']
        }
        
        # Score each category
        category_scores = {}
        for category, cat_keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if any(cat_word in keyword.lower() for cat_word in cat_keywords))
            category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                return best_category[0]
        
        return TopicCategory.COMPANY_STRATEGY  # Default category
    
    def _determine_primary_topic(self, topics: List[TopicAnalysis]) -> Tuple[str, float]:
        """Determine primary topic and confidence"""
        if not topics:
            return "general", 0.0
        
        # Sort by weight and return top topic
        sorted_topics = sorted(topics, key=lambda x: x.weight, reverse=True)
        primary_topic = sorted_topics[0]
        
        # Calculate confidence based on topic separation
        if len(sorted_topics) > 1:
            confidence = primary_topic.weight / (primary_topic.weight + sorted_topics[1].weight)
        else:
            confidence = 1.0
        
        return primary_topic.category.value, confidence
    
    async def _detect_market_signals(self, text: str, symbol: str) -> List[MarketSignal]:
        """Detect actionable market signals from text"""
        signals = []
        
        # Price target patterns
        price_targets = self._extract_price_targets(text)
        for target in price_targets:
            signal = MarketSignal(
                signal_type="price_target",
                strength=MarketSignalStrength.MODERATE,
                direction="bullish" if target > 0 else "bearish",
                confidence=0.7,
                time_horizon="short_term",
                reasoning=f"Price target mentioned: {target}",
                extracted_values={"price_target": target},
                relevant_entities=[symbol]
            )
            signals.append(signal)
        
        # Earnings signals
        earnings_signals = self._detect_earnings_signals(text)
        signals.extend(earnings_signals)
        
        # Analyst rating changes
        rating_signals = self._detect_rating_changes(text)
        signals.extend(rating_signals)
        
        # Volume and momentum signals
        momentum_signals = self._detect_momentum_signals(text)
        signals.extend(momentum_signals)
        
        return signals
    
    def _extract_price_targets(self, text: str) -> List[float]:
        """Extract price targets from text"""
        price_patterns = [
            r'\$(\d+(?:\.\d{2})?)',  # $123.45
            r'(\d+(?:\.\d{2})?)\s*dollars',  # 123.45 dollars
            r'target\s+of\s+\$?(\d+(?:\.\d{2})?)',  # target of $123
            r'price\s+target\s+\$?(\d+(?:\.\d{2})?)',  # price target $123
        ]
        
        targets = []
        for pattern in price_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    price = float(match.group(1))
                    if 1 < price < 10000:  # Reasonable stock price range
                        targets.append(price)
                except (ValueError, IndexError):
                    continue
        
        return list(set(targets))  # Remove duplicates
    
    def _detect_earnings_signals(self, text: str) -> List[MarketSignal]:
        """Detect earnings-related signals"""
        signals = []
        text_lower = text.lower()
        
        # Earnings beat/miss patterns
        beat_patterns = ['beat', 'exceeded', 'outperformed', 'above expectations']
        miss_patterns = ['miss', 'below', 'disappointed', 'fell short', 'underperformed']
        
        beat_count = sum(1 for pattern in beat_patterns if pattern in text_lower)
        miss_count = sum(1 for pattern in miss_patterns if pattern in text_lower)
        
        if beat_count > miss_count and beat_count > 0:
            signal = MarketSignal(
                signal_type="earnings_beat",
                strength=MarketSignalStrength.STRONG,
                direction="bullish",
                confidence=0.8,
                time_horizon="immediate",
                reasoning="Earnings beat detected in text",
                extracted_values={"beat_indicators": beat_count},
                relevant_entities=[]
            )
            signals.append(signal)
        elif miss_count > beat_count and miss_count > 0:
            signal = MarketSignal(
                signal_type="earnings_miss",
                strength=MarketSignalStrength.STRONG,
                direction="bearish",
                confidence=0.8,
                time_horizon="immediate",
                reasoning="Earnings miss detected in text",
                extracted_values={"miss_indicators": miss_count},
                relevant_entities=[]
            )
            signals.append(signal)
        
        return signals
    
    def _detect_rating_changes(self, text: str) -> List[MarketSignal]:
        """Detect analyst rating changes"""
        signals = []
        text_lower = text.lower()
        
        upgrade_patterns = ['upgrade', 'raised', 'increased rating', 'buy']
        downgrade_patterns = ['downgrade', 'lowered', 'reduced rating', 'sell']
        
        if any(pattern in text_lower for pattern in upgrade_patterns):
            signal = MarketSignal(
                signal_type="rating_upgrade",
                strength=MarketSignalStrength.MODERATE,
                direction="bullish",
                confidence=0.7,
                time_horizon="short_term",
                reasoning="Analyst upgrade detected",
                extracted_values={},
                relevant_entities=[]
            )
            signals.append(signal)
        
        if any(pattern in text_lower for pattern in downgrade_patterns):
            signal = MarketSignal(
                signal_type="rating_downgrade",
                strength=MarketSignalStrength.MODERATE,
                direction="bearish",
                confidence=0.7,
                time_horizon="short_term",
                reasoning="Analyst downgrade detected",
                extracted_values={},
                relevant_entities=[]
            )
            signals.append(signal)
        
        return signals
    
    def _detect_momentum_signals(self, text: str) -> List[MarketSignal]:
        """Detect momentum and volume signals"""
        signals = []
        text_lower = text.lower()
        
        momentum_patterns = {
            'bullish': ['momentum', 'rally', 'surge', 'breakout', 'strong volume'],
            'bearish': ['selloff', 'correction', 'decline', 'weak volume']
        }
        
        for direction, patterns in momentum_patterns.items():
            pattern_count = sum(1 for pattern in patterns if pattern in text_lower)
            if pattern_count > 0:
                signal = MarketSignal(
                    signal_type="momentum_signal",
                    strength=MarketSignalStrength.WEAK,
                    direction=direction,
                    confidence=0.6,
                    time_horizon="immediate",
                    reasoning=f"{direction.title()} momentum patterns detected",
                    extracted_values={"pattern_count": pattern_count},
                    relevant_entities=[]
                )
                signals.append(signal)
        
        return signals
    
    def _calculate_signal_strength(self, signals: List[MarketSignal]) -> float:
        """Calculate overall signal strength"""
        if not signals:
            return 0.0
        
        strength_values = [signal.strength.value for signal in signals]
        confidence_values = [signal.confidence for signal in signals]
        
        # Weighted average of signal strengths
        weighted_strength = sum(s * c for s, c in zip(strength_values, confidence_values))
        total_confidence = sum(confidence_values)
        
        if total_confidence > 0:
            return min(weighted_strength / total_confidence / 5.0, 1.0)  # Normalize to 0-1
        
        return 0.0
    
    def _generate_trading_recommendation(self, signals: List[MarketSignal], sentiment: float) -> str:
        """Generate trading recommendation"""
        if not signals:
            if sentiment > 0.3:
                return "WEAK_BUY"
            elif sentiment < -0.3:
                return "WEAK_SELL"
            else:
                return "HOLD"
        
        # Count signal directions
        bullish_signals = sum(1 for s in signals if s.direction == "bullish")
        bearish_signals = sum(1 for s in signals if s.direction == "bearish")
        
        # Calculate signal confidence
        avg_confidence = np.mean([s.confidence for s in signals])
        
        if bullish_signals > bearish_signals:
            if avg_confidence > 0.7:
                return "STRONG_BUY"
            else:
                return "BUY"
        elif bearish_signals > bullish_signals:
            if avg_confidence > 0.7:
                return "STRONG_SELL"
            else:
                return "SELL"
        else:
            return "HOLD"
    
    def _calculate_urgency_score(self, text: str, entities: List[NamedEntity]) -> float:
        """Calculate urgency score based on text features"""
        urgency_indicators = [
            'breaking', 'urgent', 'immediate', 'now', 'today', 'just announced',
            'alert', 'emergency', 'critical', 'developing'
        ]
        
        text_lower = text.lower()
        urgency_count = sum(1 for indicator in urgency_indicators if indicator in text_lower)
        
        # Time-based entities increase urgency
        time_entities = sum(1 for e in entities if e.label in ['DATE', 'TIME'])
        
        # Calculate urgency (0-1 scale)
        urgency_score = min((urgency_count * 0.2) + (time_entities * 0.1), 1.0)
        
        return urgency_score
    
    def _calculate_credibility_score(self, text: str) -> float:
        """Calculate credibility score based on text characteristics"""
        # Factors that increase credibility
        credibility_factors = {
            'quotes': len(re.findall(r'"[^"]*"', text)) * 0.1,
            'numbers': len(re.findall(r'\d+', text)) * 0.05,
            'formal_language': 0.5 if any(word in text.lower() for word in 
                                        ['according to', 'reported', 'stated', 'announced']) else 0.0,
            'length': min(len(text) / 1000, 0.3)  # Longer articles tend to be more credible
        }
        
        credibility_score = sum(credibility_factors.values())
        return min(credibility_score, 1.0)
    
    async def _calculate_novelty_score(self, text: str, symbol: str) -> float:
        """Calculate novelty score (simplified implementation)"""
        # This would typically compare against historical articles
        # For now, return a baseline score based on unique terms
        
        unique_terms = set(text.lower().split())
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        unique_terms -= common_words
        
        # More unique terms = higher novelty
        novelty_score = min(len(unique_terms) / 100, 1.0)
        
        return novelty_score
    
    def _calculate_market_impact_score(self, 
                                     entities: List[NamedEntity],
                                     signals: List[MarketSignal],
                                     sentiment: float) -> float:
        """Calculate potential market impact score"""
        
        # Entity-based impact
        org_count = sum(1 for e in entities if e.label in ['ORG', 'ORGANIZATION'])
        money_count = sum(1 for e in entities if e.label in ['MONEY', 'PERCENT'])
        entity_impact = min((org_count * 0.1) + (money_count * 0.15), 0.4)
        
        # Signal-based impact
        signal_impact = len(signals) * 0.1
        
        # Sentiment-based impact
        sentiment_impact = abs(sentiment) * 0.3
        
        total_impact = entity_impact + signal_impact + sentiment_impact
        return min(total_impact, 1.0)

class AdvancedNLPAgent:
    """Advanced NLP Agent for comprehensive text analysis"""
    
    def __init__(self):
        self.processor = AdvancedNLPProcessor()
    
    async def analyze_article(self, article_text: str, symbol: str, article_id: int) -> AdvancedTextAnalysis:
        """Analyze a single article with advanced NLP"""
        return await self.processor.process_text(article_text, symbol, article_id)
    
    async def batch_analyze_articles(self, articles: List[Dict[str, Any]]) -> List[AdvancedTextAnalysis]:
        """Batch analyze multiple articles"""
        results = []
        
        for article_data in articles:
            try:
                analysis = await self.analyze_article(
                    article_data['content'],
                    article_data['symbol'],
                    article_data.get('id', 0)
                )
                results.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze article {article_data.get('id')}: {e}")
                continue
        
        return results
    
    def get_analysis_summary(self, analyses: List[AdvancedTextAnalysis]) -> Dict[str, Any]:
        """Generate summary of analysis results"""
        if not analyses:
            return {"error": "No analyses provided"}
        
        # Aggregate statistics
        total_articles = len(analyses)
        avg_sentiment = np.mean([a.composite_sentiment for a in analyses])
        avg_confidence = np.mean([a.sentiment_confidence for a in analyses])
        avg_market_impact = np.mean([a.market_impact_score for a in analyses])
        
        # Signal distribution
        all_signals = [signal for analysis in analyses for signal in analysis.market_signals]
        signal_types = Counter(signal.signal_type for signal in all_signals)
        
        # Topic distribution
        topics = Counter(analysis.primary_topic for analysis in analyses)
        
        # Entity extraction
        all_entities = [entity for analysis in analyses for entity in analysis.entities]
        entity_types = Counter(entity.label for entity in all_entities)
        
        return {
            "summary_statistics": {
                "total_articles": total_articles,
                "average_sentiment": avg_sentiment,
                "average_confidence": avg_confidence,
                "average_market_impact": avg_market_impact,
            },
            "signal_distribution": dict(signal_types),
            "topic_distribution": dict(topics),
            "entity_distribution": dict(entity_types),
            "top_entities": Counter(entity.text for entity in all_entities).most_common(10),
            "market_signals_count": len(all_signals),
            "bullish_signals": sum(1 for s in all_signals if s.direction == "bullish"),
            "bearish_signals": sum(1 for s in all_signals if s.direction == "bearish")
        }

# Factory function
async def create_advanced_nlp_agent() -> AdvancedNLPAgent:
    """Create and initialize Advanced NLP Agent"""
    agent = AdvancedNLPAgent()
    logger.info("Advanced NLP Agent created successfully")
    return agent