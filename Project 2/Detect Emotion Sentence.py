import re
from collections import Counter
import textwrap
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import numpy as np
import sys
import subprocess

def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded"""
    required_resources = [
        'vader_lexicon',
        'punkt',
        'punkt_tab',
        'wordnet',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]
    
    all_downloaded = True
    
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else 
                          f'taggers/{resource}' if 'perceptron' in resource else
                          f'corpora/{resource}' if resource == 'wordnet' else
                          f'sentiment/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            try:
                nltk.download(resource, quiet=True)
                print(f"✓ {resource} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {resource}: {e}")
                all_downloaded = False
    
    return all_downloaded

class EmotionBasedReadingAnalyzer:
    def __init__(self):
        # Initialize VADER sentiment analyzer
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Error initializing sentiment analyzer: {e}")
            raise
        
        # Emotion categories based on VADER compound score ranges
        self.emotion_thresholds = {
            'very_positive': (0.7, 1.0),
            'positive': (0.3, 0.7),
            'slightly_positive': (0.1, 0.3),
            'neutral': (-0.1, 0.1),
            'slightly_negative': (-0.3, -0.1),
            'negative': (-0.7, -0.3),
            'very_negative': (-1.0, -0.7)
        }
        
        # Reading pace mapping based on emotional intensity and sentiment
        self.pace_profiles = {
            'very_positive': {
                'base_wpm_range': (180, 210),
                'pace_styles': ['Energetic and enthusiastic', 'Bursting with excitement', 'Rapid and joyful'],
                'descriptions': [
                    'Fast-paced delivery with rising intonation and emphasis',
                    'Quick, bouncy rhythm with heightened emotional peaks',
                    'Accelerated tempo reflecting intense positive emotion'
                ]
            },
            'positive': {
                'base_wpm_range': (160, 190),
                'pace_styles': ['Cheerful and upbeat', 'Warm and engaging', 'Bright and lively'],
                'descriptions': [
                    'Lively pace with pleasant, melodic flow',
                    'Engaging tempo with natural warmth and enthusiasm',
                    'Balanced speed with optimistic undertones'
                ]
            },
            'slightly_positive': {
                'base_wpm_range': (145, 165),
                'pace_styles': ['Light and pleasant', 'Mildly upbeat', 'Gently optimistic'],
                'descriptions': [
                    'Comfortable pace with subtle positive energy',
                    'Relaxed but engaged tempo with mild enthusiasm',
                    'Natural flow with gentle, hopeful nuances'
                ]
            },
            'neutral': {
                'base_wpm_range': (135, 155),
                'pace_styles': ['Balanced and measured', 'Steady and neutral', 'Even and controlled'],
                'descriptions': [
                    'Standard conversational pace with natural rhythm',
                    'Consistent tempo without emotional extremes',
                    'Balanced delivery suitable for factual content'
                ]
            },
            'slightly_negative': {
                'base_wpm_range': (125, 145),
                'pace_styles': ['Slightly subdued', 'Mildly melancholic', 'Gently serious'],
                'descriptions': [
                    'Slightly slower pace with subtle gravity',
                    'Measured tempo with mild contemplative quality',
                    'Controlled delivery with hints of seriousness'
                ]
            },
            'negative': {
                'base_wpm_range': (110, 135),
                'pace_styles': ['Somber and heavy', 'Melancholic and slow', 'Gloomy and deliberate'],
                'descriptions': [
                    'Slower pace with emotional weight and depth',
                    'Drawn-out delivery reflecting sadness or concern',
                    'Heavy tempo with downward emotional inflection'
                ]
            },
            'very_negative': {
                'base_wpm_range': (90, 120),
                'pace_styles': ['Intensely emotional', 'Deeply somber', 'Dramatically slow'],
                'descriptions': [
                    'Very slow, heavy delivery with intense emotional charge',
                    'Dramatically paced with profound emotional depth',
                    'Extremely measured tempo reflecting intense negativity'
                ]
            }
        }
    
    def get_word_emotional_intensity(self, word):
        """Get emotional intensity of a word using WordNet and VADER"""
        try:
            # Get VADER score for individual word
            word_score = self.sia.polarity_scores(word)
            
            # Get synonyms from WordNet to understand word associations
            synsets = wn.synsets(word)
            emotional_weight = 0
            arousal_level = 0
            
            for synset in synsets[:2]:  # Limit to first 2 synsets for performance
                # Check if word has emotional connotations based on related words
                for lemma in synset.lemmas()[:3]:
                    lemma_score = self.sia.polarity_scores(lemma.name().replace('_', ' '))
                    emotional_weight += abs(lemma_score['compound'])
                    arousal_level += abs(lemma_score['compound']) * 2 if abs(lemma_score['compound']) > 0.5 else abs(lemma_score['compound'])
            
            return {
                'compound': word_score['compound'],
                'emotional_weight': emotional_weight,
                'arousal': arousal_level
            }
        except:
            return {'compound': 0, 'emotional_weight': 0, 'arousal': 0}
    
    def simple_tokenize(self, text):
        """Simple word tokenization as fallback"""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def simple_pos_tag(self, words):
        """Simple POS tagging as fallback"""
        # Basic POS tag approximation
        pos_tags = []
        for word in words:
            if word.lower() in ['the', 'a', 'an']:
                pos_tags.append((word, 'DT'))
            elif word.lower() in ['is', 'am', 'are', 'was', 'were', 'be', 'been', 'being']:
                pos_tags.append((word, 'VB'))
            elif word.endswith('ing'):
                pos_tags.append((word, 'VBG'))
            elif word.endswith('ed'):
                pos_tags.append((word, 'VBD'))
            elif word.endswith('ly'):
                pos_tags.append((word, 'RB'))
            elif word.endswith('tion') or word.endswith('ment'):
                pos_tags.append((word, 'NN'))
            elif word[0].isupper():
                pos_tags.append((word, 'NNP'))
            else:
                pos_tags.append((word, 'NN'))  # Default to noun
        return pos_tags
    
    def analyze_punctuation_impact(self, sentence):
        """Analyze how punctuation affects the emotional tone and pacing"""
        punct_analysis = {
            'exclamation': {'count': 0, 'intensity_multiplier': 1.2},
            'question': {'count': 0, 'intensity_multiplier': 1.1},
            'ellipsis': {'count': 0, 'intensity_multiplier': 0.9},
            'comma': {'count': 0, 'pause_factor': 0.1},
            'period': {'count': 0, 'pause_factor': 0.2},
            'semicolon': {'count': 0, 'pause_factor': 0.15},
            'colon': {'count': 0, 'pause_factor': 0.15},
            'dash': {'count': 0, 'pause_factor': 0.12}
        }
        
        # Count punctuation
        punct_analysis['exclamation']['count'] = sentence.count('!')
        punct_analysis['question']['count'] = sentence.count('?')
        punct_analysis['ellipsis']['count'] = sentence.count('...')
        punct_analysis['comma']['count'] = sentence.count(',')
        punct_analysis['period']['count'] = sentence.count('.')
        punct_analysis['semicolon']['count'] = sentence.count(';')
        punct_analysis['colon']['count'] = sentence.count(':')
        punct_analysis['dash']['count'] = sentence.count('—') + sentence.count('-')
        
        return punct_analysis
    
    def detect_speech_patterns(self, sentence, words):
        """Detect speech patterns that affect reading pace"""
        # Use simple tokenization if NLTK tokenization fails
        try:
            nltk_words = word_tokenize(sentence)
            pos_tags = nltk.pos_tag(nltk_words)
        except:
            nltk_words = self.simple_tokenize(sentence)
            pos_tags = self.simple_pos_tag(nltk_words)
        
        patterns = {
            'imperative': False,  # Commands
            'interrogative': False,  # Questions
            'exclamatory': False,  # Exclamations
            'conditional': False,  # If/then statements
            'comparative': False,  # Comparisons
            'superlative': False,  # Superlatives
            'repetition': False,  # Repeated words
            'alliteration': False,  # Alliteration
            'complexity_score': 0  # Syntactic complexity
        }
        
        # Check for imperative mood (verbs at start)
        if words and words[0].lower() in ['go', 'stop', 'run', 'help', 'listen', 'look', 'tell', 'give', 'take', 'make', 'do']:
            patterns['imperative'] = True
        
        # Check for interrogative
        if '?' in sentence or (words and words[0].lower() in ['what', 'why', 'how', 'when', 'where', 'who', 'is', 'are', 'can', 'could', 'would', 'will', 'do', 'does', 'did']):
            patterns['interrogative'] = True
        
        # Check for exclamatory
        if '!' in sentence:
            patterns['exclamatory'] = True
        
        # Check for conditionals
        conditional_words = ['if', 'unless', 'provided', 'assuming', 'whenever']
        if any(word.lower() in conditional_words for word in words):
            patterns['conditional'] = True
        
        # Check for comparatives/superlatives
        for word in words:
            if word.endswith('er') or word.endswith('est'):
                if word.endswith('est'):
                    patterns['superlative'] = True
                else:
                    patterns['comparative'] = True
            if word.lower() in ['more', 'most', 'less', 'least']:
                if word.lower() in ['most', 'least']:
                    patterns['superlative'] = True
                else:
                    patterns['comparative'] = True
        
        # Check for repetition
        word_counts = Counter([w.lower() for w in words])
        if any(count > 2 for count in word_counts.values()):
            patterns['repetition'] = True
        
        # Check for alliteration
        if len(words) >= 3:
            for i in range(len(words)-2):
                if words[i] and words[i+1] and words[i+2]:  # Ensure non-empty strings
                    if len(words[i]) > 0 and len(words[i+1]) > 0 and len(words[i+2]) > 0:
                        if words[i][0].lower() == words[i+1][0].lower() == words[i+2][0].lower():
                            patterns['alliteration'] = True
                            break
        
        # Calculate syntactic complexity
        complexity_indicators = sum([
            len(words) > 15,  # Long sentences
            len(words) > 25,  # Very long sentences
            any(word in ['however', 'therefore', 'moreover', 'nevertheless', 'furthermore'] for word in [w.lower() for w in words]),  # Complex transitions
            len(set(words)) / len(words) > 0.8 if len(words) > 0 else 0  # Lexical diversity
        ])
        patterns['complexity_score'] = complexity_indicators / 4  # Normalize to 0-1
        
        return patterns
    
    def calculate_emotional_arousal(self, sentence):
        """Calculate emotional arousal level using VADER and word analysis"""
        # Get simple word list (always works)
        simple_words = self.simple_tokenize(sentence)
        
        # Try NLTK tokenization, fall back to simple tokenization
        try:
            words = word_tokenize(sentence)
        except:
            words = simple_words
        
        # Get overall sentiment scores
        sentiment_scores = self.sia.polarity_scores(sentence)
        
        # Analyze individual word intensities
        word_intensities = []
        arousal_scores = []
        
        for word in words:
            if word.isalpha() and len(word) > 2:  # Skip short words and punctuation
                word_analysis = self.get_word_emotional_intensity(word)
                if abs(word_analysis['compound']) > 0.2:  # Only consider emotionally charged words
                    word_intensities.append(abs(word_analysis['compound']))
                    arousal_scores.append(word_analysis['arousal'])
        
        # Calculate aggregated metrics
        avg_intensity = np.mean(word_intensities) if word_intensities else 0
        avg_arousal = np.mean(arousal_scores) if arousal_scores else 0
        
        # Calculate emotional variability (standard deviation of intensities)
        emotional_variability = np.std(word_intensities) if len(word_intensities) > 1 else 0
        
        return {
            'compound_sentiment': sentiment_scores['compound'],
            'positive_score': sentiment_scores['pos'],
            'negative_score': sentiment_scores['neg'],
            'neutral_score': sentiment_scores['neu'],
            'avg_intensity': avg_intensity,
            'avg_arousal': avg_arousal,
            'emotional_variability': emotional_variability,
            'word_count': len(simple_words),
            'charged_words': len(word_intensities)
        }
    
    def classify_emotional_category(self, compound_score):
        """Classify the emotional category based on compound score"""
        for category, (min_val, max_val) in self.emotion_thresholds.items():
            if min_val <= compound_score <= max_val:
                return category
        return 'neutral'
    
    def get_reading_frequency(self, sentence):
        """Determine optimal reading frequency based on comprehensive analysis"""
        # Simple word list for basic operations
        simple_words = self.simple_tokenize(sentence)
        
        # Analyze emotional content
        emotion_data = self.calculate_emotional_arousal(sentence)
        punct_analysis = self.analyze_punctuation_impact(sentence)
        speech_patterns = self.detect_speech_patterns(sentence, simple_words)
        
        # Classify emotional category
        emotional_category = self.classify_emotional_category(emotion_data['compound_sentiment'])
        pace_profile = self.pace_profiles[emotional_category]
        
        # Calculate base WPM from range
        base_wpm_range = pace_profile['base_wpm_range']
        
        # Adjust based on emotional intensity
        intensity_factor = emotion_data['avg_intensity']
        if intensity_factor > 0.5:
            wpm_adjustment = 10 * intensity_factor  # More intense = faster
        else:
            wpm_adjustment = 0
        
        # Adjust based on arousal
        arousal_factor = emotion_data['avg_arousal']
        wpm_adjustment += 5 * arousal_factor
        
        # Adjust based on emotional variability
        if emotion_data['emotional_variability'] > 0.3:
            wpm_adjustment += 15  # More variability = more dynamic pace
        
        # Adjust for speech patterns
        if speech_patterns['imperative']:
            wpm_adjustment += 10  # Commands are typically faster
        if speech_patterns['interrogative']:
            wpm_adjustment -= 5  # Questions slightly slower
        if speech_patterns['exclamatory']:
            wpm_adjustment += 15  # Exclamations are faster
        if speech_patterns['complexity_score'] > 0.5:
            wpm_adjustment -= 10  # Complex sentences slower
        
        # Adjust for punctuation pauses
        total_pauses = (
            punct_analysis['comma']['count'] * punct_analysis['comma']['pause_factor'] +
            punct_analysis['period']['count'] * punct_analysis['period']['pause_factor'] +
            punct_analysis['semicolon']['count'] * punct_analysis['semicolon']['pause_factor'] +
            punct_analysis['colon']['count'] * punct_analysis['colon']['pause_factor'] +
            punct_analysis['dash']['count'] * punct_analysis['dash']['pause_factor']
        )
        wpm_adjustment -= total_pauses * 20
        
        # Calculate final WPM
        base_wpm = np.mean(base_wpm_range)
        adjusted_wpm = base_wpm + wpm_adjustment
        
        # Ensure WPM stays within reasonable bounds
        adjusted_wpm = max(70, min(250, adjusted_wpm))
        
        # Calculate estimated reading time
        reading_time_seconds = (emotion_data['word_count'] / adjusted_wpm) * 60
        
        # Select appropriate pace style and description
        import random
        random.seed(hash(sentence) % 1000)  # Consistent selection for same sentence
        
        pace_style = random.choice(pace_profile['pace_styles'])
        description = random.choice(pace_profile['descriptions'])
        
        return {
            'emotional_category': emotional_category,
            'base_wpm': round(base_wpm),
            'adjusted_wpm': round(adjusted_wpm),
            'pace_style': pace_style,
            'description': description,
            'estimated_time': round(reading_time_seconds, 2),
            'word_count': emotion_data['word_count'],
            'sentiment_scores': {
                'compound': emotion_data['compound_sentiment'],
                'positive': emotion_data['positive_score'],
                'negative': emotion_data['negative_score'],
                'neutral': emotion_data['neutral_score']
            },
            'emotional_metrics': {
                'intensity': emotion_data['avg_intensity'],
                'arousal': emotion_data['avg_arousal'],
                'variability': emotion_data['emotional_variability'],
                'charged_words': emotion_data['charged_words']
            },
            'speech_patterns': speech_patterns,
            'punctuation_analysis': punct_analysis
        }
    
    def visualize_pacing(self, frequency_data):
        """Create a visual representation of reading pace"""
        wpm = frequency_data['adjusted_wpm']
        max_wpm = 250
        min_wpm = 70
        
        # Normalize to 0-30 scale for visualization
        normalized = (wpm - min_wpm) / (max_wpm - min_wpm)
        bar_length = int(normalized * 30)
        bar_length = max(1, min(30, bar_length))
        
        bar = '█' * bar_length + '░' * (30 - bar_length)
        
        # Add pace indicators
        if wpm < 100:
            pace_indicator = "🐢 Very Slow"
        elif wpm < 120:
            pace_indicator = "🐌 Slow"
        elif wpm < 140:
            pace_indicator = "🚶 Moderate-Slow"
        elif wpm < 160:
            pace_indicator = "🚶 Moderate"
        elif wpm < 180:
            pace_indicator = "🏃 Moderate-Fast"
        elif wpm < 200:
            pace_indicator = "🏃 Fast"
        else:
            pace_indicator = "⚡ Very Fast"
        
        return f"Pace: [{bar}] {wpm} WPM {pace_indicator}"

def install_requirements():
    """Install required packages"""
    required_packages = ['textblob', 'nltk', 'numpy']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed")

def main():
    # Install requirements first
    install_requirements()
    
    print("=" * 70)
    print("📖 ADVANCED EMOTION-BASED READING FREQUENCY ANALYZER")
    print("=" * 70)
    print("\nThis analyzer uses advanced NLP to detect emotional content")
    print("and suggest optimal reading frequency for ANY sentence.")
    print("\nFeatures:")
    print("  • VADER Sentiment Analysis")
    print("  • WordNet Emotional Intensity")
    print("  • Speech Pattern Detection")
    print("  • Punctuation Impact Analysis")
    print("\nType 'quit' or 'exit' to end the program.\n")
    
    # Initialize NLP components
    print("Initializing NLP components...")
    
    # Download required NLTK data
    if not ensure_nltk_resources():
        print("⚠️  Some NLTK resources could not be downloaded.")
        print("The analyzer will use fallback methods where needed.")
    
    # Initialize analyzer
    try:
        analyzer = EmotionBasedReadingAnalyzer()
        print("Ready! ✓\n")
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        return
    
    while True:
        print("-" * 70)
        user_input = input("📝 Enter your sentence: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye! Thanks for using the analyzer.")
            break
        
        if not user_input:
            print("❌ Please enter a sentence to analyze.")
            continue
        
        if len(user_input.split()) < 2:
            print("⚠️  Please enter at least 2 words for better analysis.")
            continue
        
        # Analyze the sentence
        print("\n🔍 Performing comprehensive emotional analysis...")
        try:
            frequency = analyzer.get_reading_frequency(user_input)
        except Exception as e:
            print(f"❌ Error analyzing sentence: {e}")
            continue
        
        # Display results
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE ANALYSIS RESULTS")
        print("=" * 70)
        
        print(f"\n📄 Sentence: \"{user_input}\"")
        print(f"📏 Word Count: {frequency['word_count']}")
        
        # Emotional category
        print(f"\n🎭 EMOTIONAL CATEGORY: {frequency['emotional_category'].replace('_', ' ').upper()}")
        
        # Sentiment scores
        sentiment = frequency['sentiment_scores']
        print(f"\n💭 SENTIMENT ANALYSIS:")
        print(f"   • Compound Score: {sentiment['compound']:.3f}")
        print(f"   • Positive: {sentiment['positive']:.1%}")
        print(f"   • Negative: {sentiment['negative']:.1%}")
        print(f"   • Neutral: {sentiment['neutral']:.1%}")
        
        # Emotional metrics
        metrics = frequency['emotional_metrics']
        print(f"\n📈 EMOTIONAL METRICS:")
        print(f"   • Intensity: {metrics['intensity']:.3f}")
        print(f"   • Arousal: {metrics['arousal']:.3f}")
        print(f"   • Variability: {metrics['variability']:.3f}")
        print(f"   • Emotionally Charged Words: {metrics['charged_words']}")
        
        # Reading frequency
        print(f"\n📖 READING FREQUENCY ANALYSIS:")
        print(f"   • Base WPM: {frequency['base_wpm']}")
        print(f"   • Adjusted WPM: {frequency['adjusted_wpm']}")
        print(f"   • Pacing Style: {frequency['pace_style']}")
        print(f"   • Description: {frequency['description']}")
        print(f"   • Est. Reading Time: {frequency['estimated_time']} seconds")
        
        # Visual representation
        print(f"\n{analyzer.visualize_pacing(frequency)}")
        
        # Speech patterns
        patterns = frequency['speech_patterns']
        active_patterns = [k for k, v in patterns.items() if v and k != 'complexity_score']
        if active_patterns:
            print(f"\n🗣️  DETECTED SPEECH PATTERNS:")
            for pattern in active_patterns:
                print(f"   • {pattern.replace('_', ' ').title()}")
            print(f"   • Syntactic Complexity: {patterns['complexity_score']:.2f}")
        
        # Punctuation impact
        punct = frequency['punctuation_analysis']
        total_punct = sum(v['count'] for v in punct.values())
        if total_punct > 0:
            print(f"\n❗ PUNCTUATION IMPACT:")
            for punct_type, data in punct.items():
                if data['count'] > 0:
                    print(f"   • {punct_type.title()}: {data['count']}")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
