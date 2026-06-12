# SentimentPulse: LLM-Powered Market Sentiment Analysis

## Product Overview

**SentimentPulse** is the AI/LLM component of the FIREKit ecosystem, providing real-time sentiment analysis from news, earnings calls, SEC filings, and social media. It leverages both proprietary LLMs (GPT-4, Claude) and open-source alternatives (FinGPT, FinBERT) for cost-effective sentiment extraction.

### Key Value Propositions

- **Multi-Source Analysis**: News, earnings calls, SEC filings, social media
- **Financial LLM Stack**: FinBERT for classification, FinGPT for generation
- **Real-Time Processing**: Stream news and generate signals in <5 seconds
- **Cost Optimization**: Intelligent routing between proprietary and open-source models
- **Interpretable Signals**: Explain why sentiment is positive/negative

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SentimentPulse                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Data Ingestion                            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │   News   │  │ Earnings │  │   SEC    │  │  Social  │     │    │
│  │  │   APIs   │  │  Calls   │  │ Filings  │  │  Media   │     │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │    │
│  └───────┴─────────────┴─────────────┴─────────────┴───────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                   Text Processing                            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │    │
│  │  │   NER    │  │ Chunking │  │ Cleaning │                   │    │
│  │  └──────────┘  └──────────┘  └──────────┘                   │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                   Model Router                               │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │ FinBERT  │  │  FinGPT  │  │  GPT-4   │  │  Claude  │     │    │
│  │  │  (Fast)  │  │ (Medium) │  │ (Complex)│  │(Reasoning)│     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                   Signal Generation                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │    │
│  │  │Sentiment │  │  Event   │  │ Novelty  │                   │    │
│  │  │  Score   │  │Detection │  │  Score   │                   │    │
│  │  └──────────┘  └──────────┘  └──────────┘                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Comparison

| Model | Cost | Latency | Accuracy | Best For |
|-------|------|---------|----------|----------|
| FinBERT | Free | ~10ms | 85% | High-volume classification |
| FinGPT | ~$0.001/call | ~100ms | 88% | Nuanced analysis |
| GPT-4 | ~$0.03/call | ~2s | 92% | Complex reasoning |
| Claude | ~$0.02/call | ~1.5s | 91% | Long document analysis |

## Technical Specification

### Data Ingestion

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, AsyncIterator
import aiohttp
import asyncio

@dataclass
class NewsArticle:
    """Standardized news article format."""
    id: str
    symbol: str
    headline: str
    summary: str
    content: str
    source: str
    published_at: datetime
    url: str

@dataclass
class EarningsTranscript:
    """Earnings call transcript."""
    symbol: str
    quarter: str
    date: datetime
    participants: List[str]
    sections: dict  # prepared_remarks, q_and_a, etc.
    full_text: str

@dataclass
class SECFiling:
    """SEC filing document."""
    symbol: str
    form_type: str  # 10-K, 10-Q, 8-K, etc.
    filed_date: datetime
    sections: dict
    full_text: str


class NewsIngester(ABC):
    """Abstract base for news data sources."""

    @abstractmethod
    async def get_news(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime
    ) -> List[NewsArticle]:
        pass

    @abstractmethod
    async def stream_news(
        self,
        symbols: List[str]
    ) -> AsyncIterator[NewsArticle]:
        pass


class BenzingaIngester(NewsIngester):
    """Benzinga news API integration."""

    BASE_URL = "https://api.benzinga.com/api/v2"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_news(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime
    ) -> List[NewsArticle]:
        async with aiohttp.ClientSession() as session:
            url = f"{self.BASE_URL}/news"
            params = {
                'token': self.api_key,
                'tickers': ','.join(symbols),
                'dateFrom': start.strftime('%Y-%m-%d'),
                'dateTo': end.strftime('%Y-%m-%d'),
                'pageSize': 100
            }

            async with session.get(url, params=params) as resp:
                data = await resp.json()
                return [
                    NewsArticle(
                        id=article['id'],
                        symbol=symbols[0],  # Simplified
                        headline=article['title'],
                        summary=article.get('teaser', ''),
                        content=article.get('body', ''),
                        source='benzinga',
                        published_at=datetime.fromisoformat(article['created']),
                        url=article['url']
                    )
                    for article in data
                ]


class PolygonNewsIngester(NewsIngester):
    """Polygon.io news API integration."""

    BASE_URL = "https://api.polygon.io/v2"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_news(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime
    ) -> List[NewsArticle]:
        async with aiohttp.ClientSession() as session:
            articles = []

            for symbol in symbols:
                url = f"{self.BASE_URL}/reference/news"
                params = {
                    'ticker': symbol,
                    'published_utc.gte': start.isoformat(),
                    'published_utc.lte': end.isoformat(),
                    'limit': 100,
                    'apiKey': self.api_key
                }

                async with session.get(url, params=params) as resp:
                    data = await resp.json()

                    for article in data.get('results', []):
                        articles.append(NewsArticle(
                            id=article['id'],
                            symbol=symbol,
                            headline=article['title'],
                            summary=article.get('description', ''),
                            content=article.get('article_url', ''),
                            source=article.get('publisher', {}).get('name', 'unknown'),
                            published_at=datetime.fromisoformat(
                                article['published_utc'].replace('Z', '+00:00')
                            ),
                            url=article.get('article_url', '')
                        ))

            return articles


class EarningsIngester:
    """Earnings call transcript ingestion."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_transcript(
        self,
        symbol: str,
        quarter: str  # e.g., "2024Q3"
    ) -> Optional[EarningsTranscript]:
        # Implementation for transcript API (e.g., Financial Modeling Prep, Seeking Alpha)
        pass

    def parse_transcript(self, raw_text: str) -> dict:
        """Parse transcript into sections."""
        sections = {
            'prepared_remarks': [],
            'q_and_a': [],
            'operator_notes': []
        }

        # Simple heuristic parsing
        current_section = 'prepared_remarks'
        lines = raw_text.split('\n')

        for line in lines:
            if 'question-and-answer' in line.lower() or 'q&a' in line.lower():
                current_section = 'q_and_a'
            elif 'operator' in line.lower() and ':' in line:
                sections['operator_notes'].append(line)
            else:
                sections[current_section].append(line)

        return sections
```

### Sentiment Models

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai
from anthropic import Anthropic

class FinBERTSentiment:
    """FinBERT for fast sentiment classification."""

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(device)
        self.model.eval()

        self.labels = ['negative', 'neutral', 'positive']

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of text.

        Returns:
            Dict with 'label', 'score', and 'probabilities'
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        label_idx = probs.argmax().item()

        return {
            'label': self.labels[label_idx],
            'score': probs[label_idx].item(),
            'probabilities': {
                label: probs[i].item()
                for i, label in enumerate(self.labels)
            }
        }

    def batch_analyze(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """Analyze multiple texts efficiently."""
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            for j, p in enumerate(probs):
                label_idx = p.argmax().item()
                results.append({
                    'label': self.labels[label_idx],
                    'score': p[label_idx].item(),
                    'probabilities': {
                        label: p[k].item()
                        for k, label in enumerate(self.labels)
                    }
                })

        return results


class FinGPTSentiment:
    """FinGPT for nuanced financial sentiment analysis."""

    def __init__(self, model_path: str = "FinGPT/fingpt-sentiment_llama2-13b_lora"):
        from peft import PeftModel
        from transformers import LlamaTokenizer, LlamaForCausalLM

        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
        base_model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-13b-hf",
            load_in_8bit=True,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()

    def analyze(self, text: str) -> dict:
        """Analyze sentiment with reasoning."""
        prompt = f"""Analyze the sentiment of this financial text.

Text: {text}

Provide:
1. Sentiment: positive, negative, or neutral
2. Confidence: high, medium, or low
3. Key factors driving the sentiment
4. Potential market impact

Response:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return self._parse_response(response)

    def _parse_response(self, response: str) -> dict:
        """Parse structured response from model."""
        # Extract sentiment and reasoning
        lines = response.split('\n')
        result = {
            'sentiment': 'neutral',
            'confidence': 'medium',
            'factors': [],
            'market_impact': ''
        }

        for line in lines:
            if 'sentiment:' in line.lower():
                if 'positive' in line.lower():
                    result['sentiment'] = 'positive'
                elif 'negative' in line.lower():
                    result['sentiment'] = 'negative'
            elif 'confidence:' in line.lower():
                if 'high' in line.lower():
                    result['confidence'] = 'high'
                elif 'low' in line.lower():
                    result['confidence'] = 'low'

        return result


class GPT4Sentiment:
    """GPT-4 for complex sentiment analysis requiring reasoning."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    async def analyze(self, text: str, context: str = '') -> dict:
        """Analyze sentiment with advanced reasoning."""
        system_prompt = """You are a financial sentiment analyst. Analyze the provided text and return a JSON response with:
- sentiment: "positive", "negative", or "neutral"
- confidence: 0.0 to 1.0
- summary: one sentence summary
- key_points: list of key factors affecting sentiment
- entities: mentioned companies, people, products
- market_signals: potential trading implications
- risks: identified risk factors"""

        user_prompt = f"""Context: {context}

Text to analyze:
{text}

Respond in JSON format only."""

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        import json
        return json.loads(response.choices[0].message.content)

    async def analyze_earnings_call(
        self,
        transcript: EarningsTranscript
    ) -> dict:
        """Deep analysis of earnings call transcript."""
        system_prompt = """You are an expert equity analyst. Analyze this earnings call transcript and provide:
1. Management tone assessment (confident, cautious, defensive, etc.)
2. Key guidance changes vs. prior quarter
3. Unexpected revelations or concerns
4. Analyst question sentiment and management response quality
5. Forward-looking statements analysis
6. Red flags or positive signals for investors
7. Overall investment signal: bullish, bearish, or neutral with reasoning"""

        # Process in chunks for long transcripts
        prepared = transcript.sections.get('prepared_remarks', [])
        qna = transcript.sections.get('q_and_a', [])

        prepared_analysis = await self._analyze_section(
            '\n'.join(prepared[:50]),
            "Analyze the prepared remarks section"
        )

        qna_analysis = await self._analyze_section(
            '\n'.join(qna[:50]),
            "Analyze the Q&A section"
        )

        # Synthesize
        synthesis_prompt = f"""Based on these section analyses, provide overall assessment:

Prepared Remarks: {prepared_analysis}

Q&A Session: {qna_analysis}

Provide final investment signal and key takeaways."""

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.1
        )

        return {
            'prepared_remarks_analysis': prepared_analysis,
            'qna_analysis': qna_analysis,
            'synthesis': response.choices[0].message.content
        }
```

### Model Router

```python
class ModelRouter:
    """Intelligently route requests to appropriate sentiment model."""

    def __init__(
        self,
        finbert: FinBERTSentiment,
        fingpt: Optional[FinGPTSentiment] = None,
        gpt4: Optional[GPT4Sentiment] = None
    ):
        self.finbert = finbert
        self.fingpt = fingpt
        self.gpt4 = gpt4

        # Cost per 1K tokens (approximate)
        self.costs = {
            'finbert': 0,
            'fingpt': 0.001,
            'gpt4': 0.03
        }

    async def analyze(
        self,
        text: str,
        priority: str = 'balanced',  # 'speed', 'accuracy', 'balanced', 'cost'
        text_type: str = 'news'  # 'news', 'earnings', 'filing', 'social'
    ) -> dict:
        """Route to appropriate model based on requirements."""

        # Determine complexity
        complexity = self._assess_complexity(text, text_type)

        # Select model based on priority and complexity
        if priority == 'speed' or complexity == 'simple':
            return self.finbert.analyze(text)

        elif priority == 'cost':
            if complexity in ['simple', 'moderate']:
                return self.finbert.analyze(text)
            elif self.fingpt:
                return self.fingpt.analyze(text)
            return self.finbert.analyze(text)

        elif priority == 'accuracy':
            if self.gpt4 and complexity == 'complex':
                return await self.gpt4.analyze(text)
            elif self.fingpt and complexity == 'moderate':
                return self.fingpt.analyze(text)
            return self.finbert.analyze(text)

        else:  # balanced
            if complexity == 'complex' and self.gpt4:
                return await self.gpt4.analyze(text)
            elif complexity == 'moderate' and self.fingpt:
                return self.fingpt.analyze(text)
            return self.finbert.analyze(text)

    def _assess_complexity(self, text: str, text_type: str) -> str:
        """Assess text complexity for model selection."""
        word_count = len(text.split())

        # Simple heuristics
        if text_type == 'social' or word_count < 100:
            return 'simple'
        elif text_type == 'earnings' or word_count > 1000:
            return 'complex'
        elif any(kw in text.lower() for kw in ['guidance', 'outlook', 'forecast', 'risk']):
            return 'moderate'
        else:
            return 'simple'

    def estimate_cost(self, texts: List[str], priority: str) -> float:
        """Estimate processing cost for a batch of texts."""
        total_tokens = sum(len(t.split()) * 1.3 for t in texts)  # Rough estimate

        if priority == 'speed' or priority == 'cost':
            return 0  # FinBERT is free

        elif priority == 'accuracy':
            # Assume mix of models
            return total_tokens / 1000 * 0.015

        else:
            return total_tokens / 1000 * 0.005
```

### Signal Generation

```python
class SentimentSignalGenerator:
    """Generate trading signals from sentiment analysis."""

    def __init__(self, router: ModelRouter):
        self.router = router
        self.history = {}  # Track historical sentiment

    async def generate_signal(
        self,
        symbol: str,
        articles: List[NewsArticle],
        lookback_hours: int = 24
    ) -> dict:
        """
        Generate sentiment signal for a symbol.

        Returns:
            Dict with 'signal', 'confidence', 'articles_analyzed', 'explanation'
        """
        # Analyze each article
        analyses = []
        for article in articles:
            text = f"{article.headline}\n{article.summary}"
            analysis = await self.router.analyze(
                text,
                priority='balanced',
                text_type='news'
            )
            analysis['published_at'] = article.published_at
            analysis['headline'] = article.headline
            analyses.append(analysis)

        # Compute aggregate sentiment
        positive_count = sum(1 for a in analyses if a['label'] == 'positive')
        negative_count = sum(1 for a in analyses if a['label'] == 'negative')
        total = len(analyses)

        if total == 0:
            return {
                'signal': 0,
                'confidence': 0,
                'articles_analyzed': 0,
                'explanation': 'No recent articles'
            }

        # Weighted by recency
        now = datetime.now()
        weighted_score = 0
        weight_sum = 0

        for a in analyses:
            hours_ago = (now - a['published_at']).total_seconds() / 3600
            weight = 1 / (1 + hours_ago / 12)  # Decay over 12 hours

            if a['label'] == 'positive':
                score = a.get('score', 0.7)
            elif a['label'] == 'negative':
                score = -a.get('score', 0.7)
            else:
                score = 0

            weighted_score += score * weight
            weight_sum += weight

        final_score = weighted_score / weight_sum if weight_sum > 0 else 0

        # Compute confidence based on agreement
        agreement = max(positive_count, negative_count) / total
        confidence = agreement * min(total / 5, 1)  # Scale with article count

        # Generate explanation
        if final_score > 0.3:
            signal_label = 'bullish'
        elif final_score < -0.3:
            signal_label = 'bearish'
        else:
            signal_label = 'neutral'

        top_headlines = [a['headline'] for a in sorted(
            analyses,
            key=lambda x: abs(x.get('score', 0)),
            reverse=True
        )[:3]]

        return {
            'symbol': symbol,
            'signal': final_score,  # -1 to 1
            'signal_label': signal_label,
            'confidence': confidence,
            'articles_analyzed': total,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'top_headlines': top_headlines,
            'explanation': f"{signal_label.capitalize()} sentiment based on {total} articles. "
                          f"{positive_count} positive, {negative_count} negative."
        }

    async def detect_events(
        self,
        symbol: str,
        articles: List[NewsArticle]
    ) -> List[dict]:
        """Detect significant events from news."""
        events = []

        for article in articles:
            # Check for event keywords
            keywords = {
                'earnings': ['earnings', 'revenue', 'profit', 'quarterly results'],
                'management': ['ceo', 'cfo', 'appointed', 'resigned', 'fired'],
                'merger': ['merger', 'acquisition', 'takeover', 'buyout'],
                'product': ['launch', 'release', 'announce', 'unveil'],
                'legal': ['lawsuit', 'investigation', 'settlement', 'fine'],
                'guidance': ['guidance', 'outlook', 'forecast', 'expects']
            }

            text = f"{article.headline} {article.summary}".lower()

            for event_type, kws in keywords.items():
                if any(kw in text for kw in kws):
                    events.append({
                        'type': event_type,
                        'headline': article.headline,
                        'published_at': article.published_at,
                        'source': article.source
                    })
                    break

        return events

    def compute_novelty(
        self,
        current_articles: List[NewsArticle],
        historical_articles: List[NewsArticle]
    ) -> float:
        """Compute novelty score - how different is current news from recent history."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        current_texts = [f"{a.headline} {a.summary}" for a in current_articles]
        historical_texts = [f"{a.headline} {a.summary}" for a in historical_articles]

        if not current_texts or not historical_texts:
            return 0.5  # Neutral if no comparison possible

        # Compute TF-IDF similarity
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        all_texts = current_texts + historical_texts

        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            current_vectors = tfidf_matrix[:len(current_texts)]
            historical_vectors = tfidf_matrix[len(current_texts):]

            # Average similarity to historical
            similarities = cosine_similarity(current_vectors, historical_vectors)
            avg_similarity = similarities.mean()

            # Novelty is inverse of similarity
            return 1 - avg_similarity

        except Exception:
            return 0.5
```

## Configuration

```yaml
# sentimentpulse.yaml
data_sources:
  benzinga:
    api_key: ${BENZINGA_API_KEY}
    enabled: true

  polygon:
    api_key: ${POLYGON_API_KEY}
    enabled: true

models:
  finbert:
    enabled: true
    device: cuda  # or cpu
    batch_size: 32

  fingpt:
    enabled: false  # Requires significant GPU memory
    model_path: FinGPT/fingpt-sentiment_llama2-13b_lora

  gpt4:
    enabled: true
    api_key: ${OPENAI_API_KEY}
    model: gpt-4-turbo-preview

  claude:
    enabled: false
    api_key: ${ANTHROPIC_API_KEY}

routing:
  default_priority: balanced
  cost_limit_daily: 10.0  # USD

signals:
  lookback_hours: 24
  min_articles: 3
  recency_decay_hours: 12
  event_detection: true
  novelty_scoring: true

output:
  format: json
  path: ./sentiment_signals
  realtime_webhook: null
```

## Roadmap

### v1.0 (Core)
- [x] FinBERT integration
- [x] News ingestion (Benzinga, Polygon)
- [x] Basic sentiment scoring
- [x] Model router

### v1.1 (Advanced Analysis)
- [ ] FinGPT integration
- [ ] GPT-4 for complex analysis
- [ ] Earnings call processing
- [ ] SEC filing analysis

### v1.2 (Signals)
- [ ] Event detection
- [ ] Novelty scoring
- [ ] Historical sentiment tracking
- [ ] Real-time streaming

### v2.0 (Production)
- [ ] Cost optimization
- [ ] A/B testing models
- [ ] Custom fine-tuning pipeline
- [ ] Sentiment dashboard
