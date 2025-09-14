"""
Enhanced French to LSF Translator

This application translates French text to LSF (French Sign Language) glosses with advanced linguistic analysis.
Designed for the SLOWKATHON competition celebrating multilingual creativity and cultural diversity.

Features:
- Complete verb conjugation system (130+ verbs with full conjugated forms)
- Advanced linguistic analysis with syntactic and morphological processing
- LSF canonical word ordering following sign language grammar
- Video generation pipeline for visual LSF output
- Comprehensive quality metrics and evaluation system
- Creative multilingual user interface

License: MIT
"""

# Import standard Python libraries for core functionality
import os
import json
import re
import unicodedata
import tempfile
import uuid
import shutil
import logging
import difflib
import traceback
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from enum import Enum

# Function to install required packages
def install_dependencies():
    """
    Installe automatiquement les dépendances nécessaires si elles ne sont pas présentes
    """
    import subprocess
    import sys
    
    required_packages = ['streamlit', 'moviepy', 'imageio-ffmpeg']
    
    def is_package_installed(package_name):
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False
    
    for package in required_packages:
        if not is_package_installed(package):
            print(f"Installation de {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} a été installé avec succès.")

# Installation automatique des dépendances
install_dependencies()

# Import external dependencies with graceful fallbacks for missing packages
try:
    import streamlit as st
    from moviepy import VideoFileClip, concatenate_videoclips
    import imageio_ffmpeg

    HAS_STREAMLIT = True  # Flag to check if Streamlit is available for UI
except ImportError as e:
    HAS_STREAMLIT = False
    print(f"Warning: Streamlit dependencies not available: {e}")

# ============================================================================
# LOGGING & CONFIGURATION SYSTEM
# ============================================================================

# Configure logging system for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('lsf_translator.log', encoding='utf-8')  # File output
    ]
)
logger = logging.getLogger("LSF-TRANSLATOR")


class Config:
    """Global configuration constants for the application"""
    VIDEOS_DIR = "videos"  # Directory containing LSF video files
    FALLBACK_EXTENSIONS = [".webm", ".mp4", ".mov", ".avi"]  # Supported video formats
    IGNORE_ACCENTS = True  # Whether to normalize accented characters
    DEFAULT_FPS = 30  # Frames per second for generated videos
    MAX_TEXT_LENGTH = 2000  # Maximum input text length
    CACHE_TTL = 3600  # Cache time-to-live in seconds
    MAX_VIDEO_CLIPS = 50  # Maximum number of video clips to concatenate
    MAX_CONCURRENT_CLIPS = 10  # Maximum concurrent video processing

    # UI Styling configuration for the creative interface
    MAIN_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    METRICS_GRADIENT = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
    VERB_GRADIENT = "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)"


class SessionKeys:
    """Session state key constants for Streamlit interface management"""
    TEXT = "fr_text_input_v51"
    RESULT = "translation_result_v51"
    VIDEO_BYTES = "video_bytes_v51"
    VIDEO_PATH = "video_path_v51"
    QUALITY_METRICS = "quality_metrics_v51"
    USER_FEEDBACK = "user_feedback_v51"
    TRANSLATION_HISTORY = "translation_history_v51"
    SESSION_ID = "session_id_v51"


# ============================================================================
# UTILITY FUNCTIONS FOR TEXT PROCESSING
# ============================================================================

def timing_decorator(func):
    """Decorator to measure function execution time for performance monitoring"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"{func.__name__} executed in {end - start:.3f}s")
        return result

    return wrapper


@lru_cache(maxsize=4000)
def strip_accents(text: str) -> str:
    """
    Remove accents from French text efficiently using Unicode normalization
    This helps with consistent matching between accented and non-accented forms
    """
    if not text:
        return text
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


@lru_cache(maxsize=4000)
def normalize_token(token: str, ignore_accents: bool = True) -> str:
    """
    Normalize token for consistent processing across different text inputs
    Handles case normalization, accent removal, and apostrophe standardization
    """
    if not token:
        return ""

    normalized = token.strip().lower().replace("'", "'")
    if ignore_accents:
        normalized = strip_accents(normalized)

    return normalized


def canonicalize(text: str) -> str:
    """Create canonical form of text for fuzzy matching by removing all non-alphanumeric characters"""
    normalized = normalize_token(text, Config.IGNORE_ACCENTS)
    return re.sub(r"[^a-z0-9]", "", normalized)


# ============================================================================
# VERB CONJUGATION SYSTEM - CORE LINGUISTIC ENGINE
# ============================================================================

class VerbGroup(Enum):
    """French verb groups classification for conjugation patterns"""
    FIRST = "er"  # Regular -er verbs (parler, manger, etc.)
    SECOND = "ir"  # Regular -ir verbs (finir, choisir, etc.)
    THIRD = "re"  # Regular -re verbs (vendre, attendre, etc.)
    AUXILIARY = "aux"  # Auxiliary verbs (être, avoir)
    MODAL = "modal"  # Modal verbs (pouvoir, vouloir, devoir)
    IRREGULAR = "irr"  # Irregular verbs with special conjugation patterns


class Tense(Enum):
    """French tenses supported by the conjugation system"""
    PRESENT = "présent"
    IMPERFECT = "imparfait"
    FUTURE = "futur"
    CONDITIONAL = "conditionnel"
    PAST_PARTICIPLE = "participe_passé"
    INFINITIVE = "infinitif"


class Person(Enum):
    """Grammatical persons for verb conjugation"""
    FIRST_SING = "1sg"  # je
    SECOND_SING = "2sg"  # tu
    THIRD_SING = "3sg"  # il/elle
    FIRST_PLUR = "1pl"  # nous
    SECOND_PLUR = "2pl"  # vous
    THIRD_PLUR = "3pl"  # ils/elles


@dataclass
class ConjugationForm:
    """Container for storing conjugation information with confidence scoring"""
    form: str
    infinitive: str
    tense: Tense
    person: Person
    confidence: float = 1.0


class VerbConjugationGenerator:
    """
    Comprehensive French verb conjugation generator
    Handles regular patterns and irregular verbs with complete conjugation tables
    """

    def __init__(self):
        # Regular conjugation endings for each verb group
        self.endings = {
            'er': {  # First group verbs (most common)
                'present': ['e', 'es', 'e', 'ons', 'ez', 'ent'],
                'imparfait': ['ais', 'ais', 'ait', 'ions', 'iez', 'aient'],
                'futur': ['erai', 'eras', 'era', 'erons', 'erez', 'eront'],
                'passe_compose': ['é', 'é', 'é', 'é', 'é', 'é'],
            },
            'ir': {  # Second group verbs
                'present': ['is', 'is', 'it', 'issons', 'issez', 'issent'],
                'imparfait': ['issais', 'issais', 'issait', 'issions', 'issiez', 'issaient'],
                'futur': ['irai', 'iras', 'ira', 'irons', 'irez', 'iront'],
                'passe_compose': ['i', 'i', 'i', 'i', 'i', 'i'],
            },
            're': {  # Third group verbs
                'present': ['s', 's', '', 'ons', 'ez', 'ent'],
                'imparfait': ['ais', 'ais', 'ait', 'ions', 'iez', 'aient'],
                'futur': ['rai', 'ras', 'ra', 'rons', 'rez', 'ront'],
                'passe_compose': ['u', 'u', 'u', 'u', 'u', 'u'],
            }
        }

        # Complete irregular verb conjugation table - most frequently used French verbs
        self.irregular_verbs = {
            'aller': {  # to go - highly irregular movement verb
                'present': ['vais', 'vas', 'va', 'allons', 'allez', 'vont'],
                'futur': ['irai', 'iras', 'ira', 'irons', 'irez', 'iront'],
                'passe_compose': 'allé'
            },
            'venir': {  # to come - irregular movement verb
                'present': ['viens', 'viens', 'vient', 'venons', 'venez', 'viennent'],
                'futur': ['viendrai', 'viendras', 'viendra', 'viendrons', 'viendrez', 'viendront'],
                'passe_compose': 'venu'
            },
            'voir': {  # to see - irregular perception verb
                'present': ['vois', 'vois', 'voit', 'voyons', 'voyez', 'voient'],
                'futur': ['verrai', 'verras', 'verra', 'verrons', 'verrez', 'verront'],
                'passe_compose': 'vu'
            },
            'dire': {  # to say - irregular communication verb
                'present': ['dis', 'dis', 'dit', 'disons', 'dites', 'disent'],
                'passe_compose': 'dit'
            },
            'pouvoir': {  # can/to be able - modal verb
                'present': ['peux', 'peux', 'peut', 'pouvons', 'pouvez', 'peuvent'],
                'futur': ['pourrai', 'pourras', 'pourra', 'pourrons', 'pourrez', 'pourront'],
                'passe_compose': 'pu'
            },
            'vouloir': {  # to want - modal verb
                'present': ['veux', 'veux', 'veut', 'voulons', 'voulez', 'veulent'],
                'futur': ['voudrai', 'voudras', 'voudra', 'voudrons', 'voudrez', 'voudront'],
                'passe_compose': 'voulu'
            },
            'savoir': {  # to know - irregular knowledge verb
                'present': ['sais', 'sais', 'sait', 'savons', 'savez', 'savent'],
                'futur': ['saurai', 'sauras', 'saura', 'saurons', 'saurez', 'sauront'],
                'passe_compose': 'su'
            },
            'être': {  # to be - most irregular auxiliary verb
                'present': ['suis', 'es', 'est', 'sommes', 'êtes', 'sont'],
                'imparfait': ['étais', 'étais', 'était', 'étions', 'étiez', 'étaient'],
                'futur': ['serai', 'seras', 'sera', 'serons', 'serez', 'seront'],
                'passe_compose': 'été'
            },
            'avoir': {  # to have - irregular auxiliary verb
                'present': ['ai', 'as', 'a', 'avons', 'avez', 'ont'],
                'imparfait': ['avais', 'avais', 'avait', 'avions', 'aviez', 'avaient'],
                'futur': ['aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront'],
                'passe_compose': 'eu'
            },
            'mettre': {  # to put - irregular action verb
                'present': ['mets', 'mets', 'met', 'mettons', 'mettez', 'mettent'],
                'passe_compose': 'mis'
            },
            'prendre': {  # to take - irregular action verb
                'present': ['prends', 'prends', 'prend', 'prenons', 'prenez', 'prennent'],
                'passe_compose': 'pris'
            },
            'croire': {  # to believe - irregular mental verb
                'present': ['crois', 'crois', 'croit', 'croyons', 'croyez', 'croient'],
                'passe_compose': 'cru'
            },
            'écrire': {  # to write - irregular communication verb
                'present': ['écris', 'écris', 'écrit', 'écrivons', 'écrivez', 'écrivent'],
                'passe_compose': 'écrit'
            },
            'offrir': {  # to offer - irregular action verb with -er endings
                'present': ['offre', 'offres', 'offre', 'offrons', 'offrez', 'offrent'],
                'passe_compose': 'offert'
            }
        }

    def conjugate_verb(self, infinitive: str, is_reflexive: bool = False) -> Dict[str, str]:
        """
        Generate all conjugations for a verb including reflexive forms
        Returns a dictionary mapping all conjugated forms to the base video name
        """
        conjugations = {}
        infinitive_clean = infinitive.replace("se ", "").replace("s'", "")
        video_name = infinitive.replace(" ", "_").replace("'", "")

        # Add infinitive forms to the mapping
        conjugations[infinitive] = video_name
        conjugations[infinitive_clean] = video_name

        # Handle irregular verbs with special conjugation patterns
        if infinitive_clean in self.irregular_verbs:
            verb_data = self.irregular_verbs[infinitive_clean]

            # Present tense forms
            if 'present' in verb_data:
                for i, form in enumerate(verb_data['present']):
                    conjugations[form] = video_name
                    # Add reflexive forms with appropriate pronouns
                    if is_reflexive:
                        pronouns = ['me ', 'te ', 'se ', 'nous ', 'vous ', 'se ']
                        reflexive_form = pronouns[i] + form
                        conjugations[reflexive_form] = video_name

            # Past participle forms with gender/number agreement
            if 'passe_compose' in verb_data:
                pp = verb_data['passe_compose']
                conjugations[pp] = video_name
                # Add feminine and plural agreements
                for ending in ['e', 's', 'es']:
                    conjugations[pp + ending] = video_name

            # Future tense forms
            if 'futur' in verb_data:
                for form in verb_data['futur']:
                    conjugations[form] = video_name

        else:
            # Handle regular verb conjugation using pattern-based generation
            if infinitive_clean.endswith('er'):
                stem = infinitive_clean[:-2]
                group = 'er'
            elif infinitive_clean.endswith('ir'):
                stem = infinitive_clean[:-2]
                group = 'ir'
            elif infinitive_clean.endswith('re'):
                stem = infinitive_clean[:-2]
                group = 're'
            else:
                return conjugations  # Unknown verb type

            # Generate all regular forms using pattern matching
            for tense, endings in self.endings[group].items():
                if tense == 'present':
                    for i, ending in enumerate(endings):
                        form = stem + ending
                        conjugations[form] = video_name

                        # Add reflexive forms for present tense
                        if is_reflexive:
                            pronouns = ['me ', 'te ', 'se ', 'nous ', 'vous ', 'se ']
                            reflexive_form = pronouns[i] + form
                            conjugations[reflexive_form] = video_name

                elif tense == 'passe_compose':
                    # Past participle with agreement variations
                    for ending in endings:
                        pp_form = stem + ending
                        conjugations[pp_form] = video_name
                        # Add gender/number agreements for -é participles
                        if ending == 'é':
                            for extra in ['e', 's', 'es']:
                                conjugations[pp_form + extra] = video_name

                elif tense in ['imparfait', 'futur']:
                    # Imperfect and future tense generation
                    base = infinitive_clean[:-1] if tense == 'futur' else stem
                    for ending in endings:
                        conjugations[base + ending] = video_name

        # Add variations without accents for fuzzy matching
        conjugations_no_accent = {}
        for form, video in conjugations.items():
            form_no_accent = strip_accents(form)
            if form_no_accent != form:
                conjugations_no_accent[form_no_accent] = video
        conjugations.update(conjugations_no_accent)

        return conjugations


def create_complete_verb_dictionary() -> Dict[str, str]:
    """
    Create comprehensive dictionary with ALL French verbs and their conjugations
    This is the core lexical resource for the translation system
    """

    generator = VerbConjugationGenerator()
    complete_dict = {}

    # Complete list of verbs organized by semantic categories for comprehensive coverage
    all_verbs = {
        # Action and movement verbs - fundamental human activities
        'abandonner': False, 'abdiquer': False, 'accompagner': False,
        'acheter': False, 'adhérer': False, 'adjuger': False,
        'administrer': False, 'adorer': False, 'aller': False,
        'aller chercher': False, 'apercevoir': False, 'appeler': False,
        'apporter': False, 'apprendre': False, 'approcher': False,
        'arrêter': False, 'attendre': False, 'avertir': False,

        # Communication verbs - essential for sign language translation
        'capter des mots': False, 'chercher': False,
        'chercher les ennuis': False, 'chercher quelqu\'un de perdu': False,
        'communiquer': False, 'comprendre': False, 'convaincre': False,
        'discuter': False, 'critiquer': False, 'demander': False,
        'dialoguer': False, 'dire': False, 'expliquer': False,
        'faxer': False, 'parler': False, 'téléphoner': False,

        # Emotional verbs - expressing feelings and states
        'aimer': False, 'croire': False, 'haïr': False,
        'pleurer': False, 'rire': False, 'sourire': False,

        # Reflexive verbs - self-directed actions (is_reflexive = True)
        'se réveiller': True, 'se lever': True, 'se coucher': True,
        'se laver': True, 'se habiller': True, 'se promener': True,
        'se moquer': True, 'se rappeler': True, 'se taire': True,
        's\'asseoir': True, 's\'allier': True,

        # Work-related verbs - professional and academic activities
        'commencer': False, 'embaucher': False, 'enseigner': False,
        'fabriquer': False, 'organiser': False, 'terminer': False,
        'travailler': False, 'finir': False,

        # Movement verbs - spatial displacement and physical actions
        'courir': False, 'danser': False, 'glisser': False,
        'marcher': False, 'nager': False, 'partir': False,
        'rouler': False, 'suivre': False, 'tomber': False,
        'venir': False, 'voler': False,

        # Daily life verbs - routine activities and basic needs
        'boire': False, 'boiter': False, 'casser': False,
        'changer': False, 'dormir': False, 'habiter': False,
        'manger': False, 'mettre': False, 'oublier': False,
        'payer': False, 'prendre': False, 'soigner': False,

        # State and perception verbs - mental processes and awareness
        'entendre': False, 'essayer': False, 'jouer': False,
        'montrer': False, 'participer': False, 'perdre': False,
        'pouvoir': False, 'préférer': False, 'présenter': False,
        'regarder': False, 'rester': False, 'savoir': False,
        'voir': False, 'vouloir': False,

        # Social interaction verbs - interpersonal activities
        'convoquer': False, 'défendre': False, 'donner': False,
        'engager': False, 'enlever': False, 'envoyer': False,
        'épouser': False, 'inviter': False, 'offrir': False,
        'protéger': False, 'rejoindre': False, 'rembourser': False,
        'rencontrer': False, 'rendre': False, 'répondre': False,

        # Learning verbs - educational and cognitive processes
        'copier': False, 'corriger': False, 'endoctriner': False,
        'enregistrer': False, 'étudier': False, 'former': False,
        'imiter': False, 'réviser': False,

        # Technical verbs - modern technology and specialized actions
        'capter': False, 'concentrer': False, 'configurer': False,
        'construire': False, 'convertir': False, 'démissionner': False,
        'falsifier': False, 'fonder': False, 'imprimer': False,
        'insérer': False, 'intervenir': False, 'investir': False,
        'localiser': False, 'naviguer': False, 'photographier': False,
        'programmer': False, 'scanner': False, 'surveiller': False,

        # Other essential verbs - miscellaneous important actions
        'divorcer': False, 'étonner': False, 'être': False, 'avoir': False,
        'fêter': False, 'moquer': False, 'prévenir': False,
        'proposer': False, 'raconter': False, 'ranger': False,
        'vaincre': False, 'vendre': False, 'visiter': False,
        'gronder': False, 'refuser': False, 'plonger': False
    }

    # Generate conjugations for each verb using the conjugation engine
    for infinitive, is_reflexive in all_verbs.items():
        verb_conjugations = generator.conjugate_verb(infinitive, is_reflexive)
        complete_dict.update(verb_conjugations)

    # Add essential non-verb words for complete grammar coverage
    complete_dict.update({
        # Personal pronouns - crucial for LSF subject marking
        'je': 'moi', 'j\'': 'moi', 'me': 'moi', 'moi': 'moi',
        'tu': 'toi', 'te': 'toi', 'toi': 'toi',
        'il': 'lui', 'lui': 'lui',
        'elle': 'elle',
        'nous': 'nous',
        'vous': 'vous',
        'ils': 'eux', 'eux': 'eux',
        'elles': 'elles',
        'on': 'nous',

        # Time expressions - LSF requires temporal marking at sentence beginning
        'aujourd\'hui': 'aujourd_hui', 'aujourd hui': 'aujourd_hui',
        'demain': 'demain',
        'hier': 'hier',
        'maintenant': 'maintenant',
        'bientôt': 'bientot', 'bientot': 'bientot',
        'tard': 'tard',
        'tôt': 'tot', 'tot': 'tot',
        'matin': 'matin',
        'soir': 'soir',
        'midi': 'midi',
        'minuit': 'minuit',

        # Common nouns - frequently referenced objects and concepts
        'travail': 'travail',
        'maison': 'maison',
        'école': 'ecole', 'ecole': 'ecole',
        'bureau': 'bureau',
        'voiture': 'voiture',
        'train': 'train',
        'bus': 'bus',
        'ami': 'ami', 'amie': 'amie',
        'famille': 'famille',
        'enfant': 'enfant',
        'homme': 'homme',
        'femme': 'femme',

        # Negation markers - LSF negation handling
        'ne': 'neg', 'pas': 'neg', 'non': 'neg',
        'jamais': 'jamais', 'rien': 'rien', 'personne': 'personne'
    })

    return complete_dict


# ============================================================================
# TEXT PROCESSING & TOKENIZATION SYSTEM
# ============================================================================

@dataclass
class Token:
    """Enhanced token with comprehensive linguistic information for analysis"""
    text: str  # Original text form
    normalized: str  # Normalized form for matching
    lemma: str  # Base form of the word
    pos: str  # Part of speech tag
    is_compound: bool = False  # Whether it's a compound word
    components: List[str] = field(default_factory=list)  # Components if compound
    confidence: float = 1.0  # Confidence score for analysis


class TextProcessor:
    """
    Advanced text processing system with French contraction expansion
    Handles the complexities of French orthography and contractions
    """

    # French contractions mapping - essential for proper tokenization
    CONTRACTIONS = {
        "j'ai": ["je", "ai"],  # I have
        "j'aime": ["je", "aime"],  # I love
        "j'arrive": ["je", "arrive"],  # I arrive
        "j'habite": ["je", "habite"],  # I live
        "j'apprends": ["je", "apprends"],  # I learn
        "c'est": ["ce", "est"],  # it is
        "n'est": ["ne", "est"],  # is not
        "n'ai": ["ne", "ai"],  # don't have
        "l'ai": ["le", "ai"],  # have it
        "l'aime": ["le", "aime"],  # love it
        "d'accord": ["de", "accord"],  # agree
        "qu'est-ce": ["quoi", "est", "ce"],  # what is
        "aujourd'hui": ["aujourd'hui"],  # today (kept as single unit)
    }

    @classmethod
    def expand_contractions(cls, text: str) -> str:
        """
        Expand French contractions for better linguistic analysis
        This preprocessing step improves tokenization accuracy
        """
        text_lower = text.lower()
        for contraction, expansion in cls.CONTRACTIONS.items():
            if contraction in text_lower:
                text_lower = text_lower.replace(contraction, " ".join(expansion))
        return text_lower

    @staticmethod
    def tokenize_pattern() -> str:
        """Return regex pattern for comprehensive French tokenization"""
        return r"([A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)*|[0-9]+|[\.!?,;:\-…—–\(\)])"


class POSTagger:
    """
    Simple but effective Part-of-Speech tagger for French text
    Provides grammatical category information for LSF processing
    """

    # French personal pronouns - essential for LSF subject identification
    PRONOUNS = {
        "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on",
        "moi", "toi", "lui", "eux", "me", "te", "se", "nous", "vous"
    }

    # French verb endings - pattern-based verb identification
    VERB_ENDINGS = {
        "er", "ir", "re", "oir", "ais", "ait", "ons", "ez", "ent",
        "ai", "as", "a", "ont", "es", "est", "sont"
    }

    @classmethod
    def get_pos(cls, token: str) -> str:
        """
        Determine part of speech for a token using heuristic rules
        Returns simplified POS tags suitable for LSF processing
        """
        normalized = normalize_token(token, True)

        if normalized in cls.PRONOUNS:
            return "PRON"  # Pronoun
        elif any(normalized.endswith(end) for end in cls.VERB_ENDINGS):
            return "VERB"  # Verb
        elif normalized.isdigit():
            return "NUM"  # Number
        elif len(normalized) == 1 and not normalized.isalnum():
            return "PUNCT"  # Punctuation
        else:
            return "NOUN"  # Default to noun for unknown words


def advanced_tokenize(text: str) -> List[Token]:
    """
    Advanced tokenization with POS tagging and confidence scoring
    This is the entry point for text processing in the translation pipeline
    """
    if not text:
        return []

    # Expand contractions and normalize whitespace
    processed_text = TextProcessor.expand_contractions(text)
    processed_text = re.sub(r"\s+", " ", processed_text.strip())

    # Apply regex tokenization pattern
    pattern = TextProcessor.tokenize_pattern()
    raw_tokens = re.findall(pattern, processed_text)

    tokens = []
    for raw_token in raw_tokens:
        if not raw_token:
            continue

        # Handle punctuation separately
        if re.fullmatch(r"[\.!?,;:\-…—–\(\)]", raw_token):
            tokens.append(Token(
                text=raw_token,
                normalized=raw_token,
                lemma=raw_token,
                pos="PUNCT"
            ))
            continue

        # Process word tokens with linguistic analysis
        normalized = normalize_token(raw_token, True)
        if not normalized:
            continue

        pos = POSTagger.get_pos(raw_token)
        lemma = normalized
        confidence = 0.9 if pos in ["PRON", "PUNCT"] else 0.7

        tokens.append(Token(
            text=raw_token,
            normalized=normalized,
            lemma=lemma,
            pos=pos,
            confidence=confidence
        ))

    return tokens


# ============================================================================
# LSF LINGUISTIC RULES AND GRAMMAR SYSTEM
# ============================================================================

class LSFRules:
    """
    LSF (French Sign Language) linguistic rules and grammatical mappings
    This class encodes the specific grammar rules of French Sign Language
    """

    # Pronoun mapping from French to LSF forms - critical for subject marking
    PRONOUN_MAPPING = {
        'je': 'moi', "j'": 'moi', 'j': 'moi',  # I -> ME
        'tu': 'toi', 'il': 'lui', 'elle': 'elle',  # you -> YOU, he -> HIM, she -> SHE
        'nous': 'nous', 'vous': 'vous',  # we -> WE, you(plural) -> YOU
        'ils': 'eux', 'elles': 'elles', 'on': 'nous',  # they -> THEM, we(informal) -> WE
        'moi': 'moi', 'toi': 'toi', 'lui': 'lui', 'eux': 'eux'  # Direct forms
    }

    # Function words typically omitted in LSF - reduces visual complexity
    FUNCTION_WORDS_TO_OMIT = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'dans',  # Articles and prepositions
        'sur', 'avec', 'sans', 'pour', 'par', 'en', 'a', 'au', 'aux',  # More prepositions
        'ce', 'cette', 'ces', 'mon', 'ma', 'ton', 'ta', 'tes', 'son',  # Demonstratives and possessives
        'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',  # More possessives
        'qui', 'que', 'dont', 'ou', 'et', 'mais', 'donc', 'car', 'ni',  # Conjunctions and relatives
        'or', 'ne', 'pas', 'deux'  # Other function words
    }

    # Temporal expressions - must appear at beginning in LSF canonical order
    TEMPORAL_EXPRESSIONS = {
        'maintenant', 'hier', 'demain', "aujourd'hui", 'matin', 'soir',
        'midi', 'tard', 'tot', 'souvent', 'jamais', 'toujours', 'parfois'
    }

    # Modal verbs - express possibility, necessity, or volition
    MODALS = {'pouvoir', 'devoir', 'vouloir'}

    # Negation markers - LSF uses specific negation strategies
    NEGATION_MARKERS = {'ne', 'pas', 'jamais', 'personne', 'rien', 'aucun'}

    # Reflexive clitics - particles indicating self-directed action
    REFLEXIVE_CLITICS = {
        "me", "m", "m'", "te", "t", "t'", "se", "s", "s'", "nous", "vous"
    }


# ============================================================================
# SYNTACTIC ANALYSIS SYSTEM
# ============================================================================

@dataclass
class SyntacticAnalysis:
    """Container for comprehensive syntactic analysis results"""
    subjects: List[Dict[str, Any]] = field(default_factory=list)  # Subject identification
    verbs: List[Dict[str, Any]] = field(default_factory=list)  # Verb analysis
    temporal_markers: List[Dict[str, Any]] = field(default_factory=list)  # Time expressions
    spatial_markers: List[Dict[str, Any]] = field(default_factory=list)  # Space expressions
    negation: bool = False  # Negation presence
    modality: List[str] = field(default_factory=list)  # Modal expressions
    question_markers: List[str] = field(default_factory=list)  # Question indicators


@dataclass
class DecisionLog:
    """
    Log entry for translation decisions - provides transparency and debugging
    Essential for understanding the translation process and quality assessment
    """
    step: str  # Processing step name
    input_data: Dict[str, Any]  # Input to this step
    output_data: Dict[str, Any]  # Output from this step
    rule_or_model: str  # Rule or model used
    timestamp: float = field(default_factory=time.time)  # When decision was made
    confidence: float = 1.0  # Confidence in the decision


@dataclass
class LSFPlan:
    """
    Comprehensive LSF planning structure for canonical ordering
    Organizes the translation according to sign language grammar
    """
    gloss_sequence: List[str] = field(default_factory=list)  # Final gloss sequence
    spatial_mapping: Dict[str, Any] = field(default_factory=dict)  # Spatial relationships
    prosody_markers: List[Dict[str, Any]] = field(default_factory=list)  # Prosodic information
    role_shifts: List[Dict[str, Any]] = field(default_factory=list)  # Perspective changes
    discourse_markers: List[str] = field(default_factory=list)  # Discourse organization
    notes: List[str] = field(default_factory=list)  # Processing notes


class SyntacticAnalyzer:
    """
    Advanced syntactic analyzer for French text processing
    Identifies grammatical structures relevant to LSF translation
    """

    def __init__(self, verb_dict: Dict[str, str]):
        self.verb_dict = verb_dict  # Reference to verb dictionary for verb identification

    @timing_decorator
    def analyze_sentence(self, tokens: List[Token]) -> SyntacticAnalysis:
        """
        Perform comprehensive syntactic analysis of tokenized sentence
        Identifies all relevant grammatical features for LSF processing
        """
        analysis = SyntacticAnalysis()

        for i, token in enumerate(tokens):
            normalized = token.normalized

            # Subject detection using pronoun mapping
            if normalized in LSFRules.PRONOUN_MAPPING:
                analysis.subjects.append({
                    'position': i,
                    'token': token.text,
                    'normalized': normalized,
                    'lsf_form': LSFRules.PRONOUN_MAPPING[normalized]
                })

            # Temporal marker detection - crucial for LSF word order
            if normalized in LSFRules.TEMPORAL_EXPRESSIONS:
                analysis.temporal_markers.append({
                    'position': i,
                    'token': token.text,
                    'normalized': normalized
                })

            # Negation detection - affects sentence-level grammar
            if normalized in LSFRules.NEGATION_MARKERS:
                analysis.negation = True

            # Modal verb detection - expresses modality
            if normalized in LSFRules.MODALS:
                analysis.modality.append(normalized)

            # Verb analysis using dictionary lookup
            if normalized in self.verb_dict:
                analysis.verbs.append({
                    'position': i,
                    'token': token.text,
                    'normalized': normalized,
                    'gloss': self.verb_dict[normalized],
                    'confidence': 0.85
                })

        return analysis


class LSFPlanner:
    """
    Advanced LSF sequence planner following canonical word order
    Implements the grammatical rules of French Sign Language
    """

    # LSF canonical order: TIME→PLACE→SUBJECT→OBJECT→VERB→MODALITY→NEGATION
    # This order reflects the visual-spatial nature of sign languages
    CANONICAL_ORDER = ['TIME', 'PLACE', 'SUBJECT', 'OBJECT', 'VERB', 'MODALITY', 'NEGATION', 'OTHER']

    @staticmethod
    def classify_gloss(gloss: str) -> str:
        """
        Classify a gloss into LSF syntactic categories for proper ordering
        Uses linguistic rules to determine grammatical function
        """
        normalized = normalize_token(gloss, True)

        # Temporal expressions have highest priority - must come first in LSF
        if normalized in LSFRules.TEMPORAL_EXPRESSIONS:
            return 'TIME'
        # Personal pronouns - subject marking in LSF
        elif normalized in {'moi', 'toi', 'lui', 'elle', 'nous', 'vous', 'eux', 'elles'}:
            return 'SUBJECT'
        # Modal verbs - express necessity, possibility, volition
        elif normalized in LSFRules.MODALS:
            return 'MODALITY'
        # Negation markers - special handling for negative constructions
        elif 'neg' in normalized or '(*)' in normalized:
            return 'NEGATION' if 'neg' in normalized else 'OTHER'
        # Regular verbs - main predicates (excluding temporal adverbs)
        elif (any(normalized.endswith(end) for end in ['er', 'ir', 're']) and
              normalized not in LSFRules.TEMPORAL_EXPRESSIONS):
            return 'VERB'
        else:
            return 'OTHER'  # Default category for other elements

    @timing_decorator
    def plan_sequence(self, glosses: List[str], has_negation: bool = False) -> Tuple[LSFPlan, List[DecisionLog]]:
        """
        Plan the LSF sequence according to canonical order
        Organizes glosses following sign language grammar rules
        """
        # Initialize category buckets for organizing glosses
        buckets = {category: [] for category in self.CANONICAL_ORDER}
        logs = []  # Decision log for transparency

        # Classify and bucket glosses according to grammatical function
        for gloss in glosses:
            category = self.classify_gloss(gloss)
            buckets[category].append(gloss)

            # Log each classification decision
            logs.append(DecisionLog(
                step='classification',
                input_data={'gloss': gloss},
                output_data={'category': category},
                rule_or_model='lsf_canonical_order',
                confidence=0.8
            ))

        # Handle negation insertion if detected but not explicitly marked
        if has_negation and not buckets['NEGATION']:
            buckets['NEGATION'].append('NEG')
            logs.append(DecisionLog(
                step='negation_insertion',
                input_data={'detected': True},
                output_data={'inserted': 'NEG'},
                rule_or_model='lsf_negation_rule',
                confidence=0.9
            ))

        # Build final sequence following canonical order
        final_sequence = []
        prosody_markers = []  # Store prosodic information

        for category in self.CANONICAL_ORDER:
            if buckets[category]:
                final_sequence.extend(buckets[category])

                # Add prosodic information for special categories
                if category == 'NEGATION':
                    prosody_markers.append({
                        'type': 'negation',
                        'scope': 'sentence',
                        'markers': buckets[category]
                    })
                elif category == 'MODALITY':
                    prosody_markers.append({
                        'type': 'modality',
                        'intensity': 'medium',
                        'markers': buckets[category]
                    })

        # Create comprehensive LSF plan
        plan = LSFPlan(
            gloss_sequence=final_sequence,
            prosody_markers=prosody_markers,
            notes=[f"Applied canonical LSF order: {' → '.join(self.CANONICAL_ORDER)}"]
        )

        return plan, logs


# ============================================================================
# QUALITY EVALUATION SYSTEM
# ============================================================================

@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics for translation assessment
    Provides multi-dimensional evaluation of translation quality
    """
    total_words: int = 0  # Total input words processed
    translated_words: int = 0  # Successfully translated words
    unknown_words: int = 0  # Unrecognized words
    ignored_words: int = 0  # Function words omitted (normal in LSF)
    confidence_score: float = 0.0  # Overall confidence in translation
    syntactic_accuracy: float = 0.0  # Grammatical structure accuracy
    morphological_accuracy: float = 0.0  # Word form accuracy
    contextual_relevance: float = 0.0  # Contextual appropriateness
    planning_quality: float = 0.0  # LSF sequence planning quality
    overall_quality: float = 0.0  # Weighted overall score


class QualityEvaluator:
    """
    Advanced quality evaluator for LSF translations
    Implements multi-criteria assessment methodology
    """

    def __init__(self):
        # Weights for different quality dimensions
        self.weights = {
            'coverage': 0.25,  # Lexical coverage
            'accuracy': 0.25,  # Translation accuracy
            'syntactic': 0.20,  # Syntactic correctness
            'morphological': 0.15,  # Morphological handling
            'planning': 0.15  # LSF planning quality
        }

    @timing_decorator
    def evaluate(self,
                 source_text: str,
                 glosses: List[str],
                 unknown_words: List[str],
                 syntactic_analysis: SyntacticAnalysis,
                 plan: Optional[LSFPlan] = None) -> QualityMetrics:
        """
        Comprehensive quality evaluation of translation results
        Analyzes multiple dimensions of translation quality
        """

        # Tokenize source text and filter out punctuation
        tokens = [t for t in advanced_tokenize(source_text) if t.pos not in ['PUNCT']]

        # Filter out function words that are normally omitted in LSF
        relevant_tokens = []
        for token in tokens:
            normalized = normalize_token(token.text, True)
            # Keep only semantically meaningful words
            if (normalized not in LSFRules.FUNCTION_WORDS_TO_OMIT or
                    normalized in LSFRules.TEMPORAL_EXPRESSIONS):
                relevant_tokens.append(token)

        total_relevant_words = len(relevant_tokens)
        translated_words = len([g for g in glosses if not g.endswith('(*)')])
        unknown_count = len(unknown_words)

        # Calculate coverage score based on meaningful words only
        coverage_score = translated_words / max(total_relevant_words, 1)

        # Calculate individual quality dimensions
        accuracy_score = self._calculate_accuracy_score(glosses)
        syntactic_score = self._calculate_syntactic_score(syntactic_analysis)
        morphological_score = self._calculate_morphological_score(syntactic_analysis)
        planning_score = self._calculate_planning_score(plan) if plan else 0.5

        # Calculate weighted overall quality score
        overall_quality = (
                coverage_score * self.weights['coverage'] +
                accuracy_score * self.weights['accuracy'] +
                syntactic_score * self.weights['syntactic'] +
                morphological_score * self.weights['morphological'] +
                planning_score * self.weights['planning']
        )

        return QualityMetrics(
            total_words=total_relevant_words,
            translated_words=translated_words,
            unknown_words=unknown_count,
            ignored_words=max(0, total_relevant_words - translated_words - unknown_count),
            confidence_score=coverage_score,
            syntactic_accuracy=syntactic_score,
            morphological_accuracy=morphological_score,
            contextual_relevance=accuracy_score,
            planning_quality=planning_score,
            overall_quality=overall_quality
        )

    def _calculate_accuracy_score(self, glosses: List[str]) -> float:
        """Calculate accuracy based on gloss quality and validity"""
        if not glosses:
            return 0.0

        valid_glosses = sum(1 for g in glosses if g and not g.endswith('(*)'))
        return valid_glosses / len(glosses)

    def _calculate_syntactic_score(self, analysis: SyntacticAnalysis) -> float:
        """Calculate syntactic accuracy score based on grammatical analysis"""
        if not analysis.verbs:
            return 0.5  # Base score when no verbs detected

        # Average confidence of verb analyses
        verb_confidences = [v.get('confidence', 0.5) for v in analysis.verbs]
        return sum(verb_confidences) / len(verb_confidences)

    def _calculate_morphological_score(self, analysis: SyntacticAnalysis) -> float:
        """Calculate morphological accuracy score based on word form analysis"""
        base_score = 0.5

        # Bonus for proper verb analysis
        if analysis.verbs:
            base_score += 0.2

        # Bonus for temporal markers (important in LSF)
        if analysis.temporal_markers:
            base_score += 0.2

        # Bonus for proper subject handling
        if analysis.subjects:
            base_score += 0.1

        return min(1.0, base_score)

    def _calculate_planning_score(self, plan: LSFPlan) -> float:
        """Calculate planning quality score based on LSF sequence organization"""
        if not plan or not plan.gloss_sequence:
            return 0.0

        score = 0.6  # Base score for having a plan

        # Bonus for prosodic markers (important for sign language)
        if plan.prosody_markers:
            score += 0.2

        # Bonus for discourse markers
        if plan.discourse_markers:
            score += 0.1

        # Bonus for reasonable sequence length
        if 2 <= len(plan.gloss_sequence) <= 20:
            score += 0.1

        return min(1.0, score)


# ============================================================================
# VIDEO PROCESSING SYSTEM
# ============================================================================

@lru_cache(maxsize=1)
def load_video_dictionary() -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Load and process the video dictionary from JSON file
    This dictionary maps French words to their corresponding LSF video files
    """
    dictionary_path = Path("data") / "dictionnaire.json"

    if dictionary_path.exists():
        try:
            with open(dictionary_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            normalized_dict = {}
            all_glosses = set()

            # Process each entry in the dictionary
            for key, value in raw_data.items():
                if not key:
                    continue

                norm_key = normalize_token(key, True)

                # Handle both string and list values
                if isinstance(value, str):
                    glosses = [normalize_token(value, True)]
                elif isinstance(value, list):
                    glosses = [normalize_token(v, True) for v in value if v]
                else:
                    continue

                if glosses:
                    normalized_dict[norm_key] = glosses
                    all_glosses.update(glosses)

            return normalized_dict, all_glosses

        except Exception as e:
            logger.error(f"Error loading dictionary: {e}")

    # Fallback to generated dictionary if file not found
    complete_dict = create_complete_verb_dictionary()
    normalized_dict = {}
    all_glosses = set()

    for key, value in complete_dict.items():
        norm_key = normalize_token(key, True)
        norm_value = normalize_token(value, True)

        if norm_key and norm_value:
            normalized_dict[norm_key] = [norm_value]
            all_glosses.add(norm_value)

    return normalized_dict, all_glosses


@lru_cache(maxsize=1)
def build_video_index(videos_dir: str) -> Dict[str, str]:
    """
    Build comprehensive video file index for LSF clips
    Maps normalized video names to file paths for efficient lookup
    """
    video_index = {}

    if not os.path.isdir(videos_dir):
        logger.warning(f"Video directory not found: {videos_dir}")
        return video_index

    try:
        # Walk through all video files in directory tree
        for root, _, files in os.walk(videos_dir):
            for file in files:
                file_lower = file.lower()
                # Check if file has supported video extension
                if not any(file_lower.endswith(ext) for ext in Config.FALLBACK_EXTENSIONS):
                    continue

                file_path = os.path.join(root, file)
                stem = os.path.splitext(file)[0]  # Remove file extension

                if not stem:
                    continue

                # Create multiple key variations for better matching
                base_key = normalize_token(stem, True)
                key_variations = {
                    base_key,
                    base_key.replace("_", " "),
                    base_key.replace(" ", "_"),
                    canonicalize(base_key)
                }

                # Add all variations to index
                for key in key_variations:
                    if key and key not in video_index:
                        video_index[key] = file_path

        logger.info(f"Video index built: {len(video_index)} entries from {videos_dir}")

    except Exception as e:
        logger.error(f"Error building video index: {e}")

    return video_index


def check_ffmpeg_availability() -> bool:
    """
    Check if FFmpeg is available for video processing
    FFmpeg is required for concatenating LSF video clips
    """
    try:
        # Check if ffmpeg is in system PATH
        if shutil.which("ffmpeg"):
            return True

        # Try to use imageio-ffmpeg as fallback
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
        return True

    except Exception as e:
        logger.warning(f"FFmpeg not available: {e}")
        return False


class VideoGenerator:
    """
    Advanced video generator with error handling and progress tracking
    Creates LSF videos by concatenating individual sign clips
    """

    def __init__(self, video_index: Dict[str, str]):
        self.video_index = video_index
        self.ffmpeg_available = check_ffmpeg_availability()

    @timing_decorator
    def generate_lsf_video(self, glosses: List[str]) -> Optional[str]:
        """
        Generate LSF video from gloss sequence by concatenating individual clips
        Returns path to generated video file or None if generation fails
        """
        if not self.ffmpeg_available:
            logger.error("FFmpeg not available for video generation")
            return None

        if not glosses:
            logger.warning("No glosses provided for video generation")
            return None

        # Filter glosses to only include those with available video clips
        filtered_glosses = self._filter_glosses(glosses)
        if not filtered_glosses:
            logger.warning("No valid video clips found for the given glosses")
            return None

        return self._create_video_sequence(filtered_glosses)

    def _filter_glosses(self, glosses: List[str]) -> List[str]:
        """Filter glosses to only include those with available video files"""
        filtered = []

        for gloss in glosses[:Config.MAX_VIDEO_CLIPS]:  # Limit number of clips
            # Skip special markers that don't have video representations
            if any(marker in gloss.lower() for marker in ['(*)', 'neg', 'negation']):
                continue

            if self._find_video_path(gloss):
                filtered.append(gloss)

        return filtered

    def _find_video_path(self, gloss: str) -> Optional[str]:
        """Find video file path for a given gloss using fuzzy matching"""
        normalized_gloss = normalize_token(gloss, True)

        # Try different key variations for robust matching
        candidates = [
            normalized_gloss,
            normalized_gloss.replace(" ", "_"),
            normalized_gloss.replace("_", " "),
            canonicalize(normalized_gloss)
        ]

        for candidate in candidates:
            if candidate in self.video_index:
                path = self.video_index[candidate]
                if os.path.exists(path):  # Verify file exists
                    return path

        return None

    def _create_video_sequence(self, glosses: List[str]) -> Optional[str]:
        """Create video sequence from filtered glosses using MoviePy"""
        clips = []
        missing_clips = []

        try:
            # Load individual video clips
            for gloss in glosses:
                video_path = self._find_video_path(gloss)
                if not video_path:
                    missing_clips.append(gloss)
                    continue

                try:
                    # Load video clip and validate
                    clip = VideoFileClip(video_path)
                    if hasattr(clip, 'duration') and clip.duration > 0:
                        clips.append(clip)
                    else:
                        missing_clips.append(gloss)
                        clip.close()
                except Exception as e:
                    logger.warning(f"Failed to load clip {video_path}: {e}")
                    missing_clips.append(gloss)

            if not clips:
                logger.error("No valid video clips could be loaded")
                return None

            # Concatenate clips into final video
            final_clip = concatenate_videoclips(clips, method="compose")

            # Generate unique output path
            temp_dir = tempfile.gettempdir()
            video_id = uuid.uuid4().hex[:8]
            mp4_path = os.path.join(temp_dir, f"lsf_translation_{video_id}.mp4")

            # Render final video with optimized settings
            final_clip.write_videofile(
                mp4_path,
                codec="libx264",  # H.264 codec for compatibility
                fps=Config.DEFAULT_FPS,  # Standard frame rate
                audio=False,  # No audio needed for sign language
                logger=None  # Suppress MoviePy logging
            )

            logger.info(f"Video generated successfully: {mp4_path}")
            return mp4_path

        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None

        finally:
            # Clean up video clips to free memory
            try:
                if 'final_clip' in locals():
                    final_clip.close()
                for clip in clips:
                    clip.close()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")


# ============================================================================
# MAIN TRANSLATOR CLASS - CORE TRANSLATION ENGINE
# ============================================================================

@dataclass
class EnhancedTranslationResult:
    """
    Comprehensive translation result with complete analysis data
    Contains all information about the translation process and results
    """
    glosses: List[str] = field(default_factory=list)  # Final LSF glosses
    unknown_words: List[str] = field(default_factory=list)  # Unrecognized words
    suggestions: Dict[str, List[str]] = field(default_factory=dict)  # Suggestions for unknowns
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)  # Quality scores
    syntactic_analysis: SyntacticAnalysis = field(default_factory=SyntacticAnalysis)  # Grammar analysis
    verb_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Verb-specific analysis
    processing_time: float = 0.0  # Processing duration
    decision_log: List[DecisionLog] = field(default_factory=list)  # Decision transparency
    plan: Optional[LSFPlan] = None  # LSF planning result
    debug_info: List[str] = field(default_factory=list)  # Debug information


class EnhancedLSFTranslator:
    """
    Advanced French to LSF translator with comprehensive linguistic analysis
    This is the main translation engine that coordinates all subsystems
    """
    video_index: dict[str, str]

    def __init__(self, dictionary: Dict[str, List[str]], video_index: Dict[str, str]):
        # Generate complete dictionary with all verb conjugations
        self.complete_conjugation_dict = create_complete_verb_dictionary()

        # Merge with existing dictionary for comprehensive coverage
        merged_dict = {}
        for key, value in self.complete_conjugation_dict.items():
            normalized_key = normalize_token(key, Config.IGNORE_ACCENTS)
            merged_dict[normalized_key] = value

        # Add dictionary entries from external source
        for key, value_list in dictionary.items():
            normalized_key = normalize_token(key, Config.IGNORE_ACCENTS)
            if normalized_key not in merged_dict and value_list:
                merged_dict[normalized_key] = value_list[0]

        self.dictionary = merged_dict
        self.video_index = video_index

        # Initialize subsystem components
        self.syntactic_analyzer = SyntacticAnalyzer(self.dictionary)
        self.quality_evaluator = QualityEvaluator()
        self.lsf_planner = LSFPlanner()

        # Exception patterns for priority matching - handles special cases
        self.exception_patterns = [
            # Reflexive patterns - HIGHEST PRIORITY for proper LSF representation
            (["me", "reveille"], "se_reveiller"),  # I wake up
            (["me", "réveille"], "se_reveiller"),  # I wake up (accented)
            (["me", "leve"], "se_lever"),  # I get up
            (["me", "lève"], "se_lever"),  # I get up (accented)
            (["me", "couche"], "se_coucher"),  # I go to bed
            (["me", "lave"], "se_laver"),  # I wash myself
            (["me", "habille"], "se_habiller"),  # I get dressed
            (["te", "reveilles"], "se_reveiller"),  # You wake up
            (["te", "réveilles"], "se_reveiller"),  # You wake up (accented)
            (["te", "leves"], "se_lever"),  # You get up
            (["te", "lèves"], "se_lever"),  # You get up (accented)
            (["se", "reveille"], "se_reveiller"),  # He/she wakes up
            (["se", "réveille"], "se_reveiller"),  # He/she wakes up (accented)
            (["se", "leve"], "se_lever"),  # He/she gets up
            (["se", "lève"], "se_lever"),  # He/she gets up (accented)

            # Common French contractions
            (["j", "ai"], "avoir"),  # I have
            (["j", "aime"], "aimer"),  # I love
            (["c", "est"], "être"),  # It is
        ]

        logger.info(f"Enhanced LSF Translator initialized with {len(self.dictionary)} entries")

    @timing_decorator
    def translate(self,
                  text: str,
                  uppercase_glosses: bool = True,
                  use_planner: bool = True) -> EnhancedTranslationResult:
        """
        Translate French text to LSF glosses with comprehensive analysis

        This is the main translation method that coordinates the entire pipeline:
        1. Input validation and preprocessing
        2. Tokenization and linguistic analysis
        3. Lexical mapping and exception pattern matching
        4. LSF canonical ordering and planning
        5. Quality evaluation and result compilation

        Args:
            text: Input French text to translate
            uppercase_glosses: Whether to output glosses in uppercase (LSF convention)
            use_planner: Whether to use LSF canonical ordering rules

        Returns:
            EnhancedTranslationResult with complete analysis and translation data
        """
        start_time = time.time()

        # Input validation - ensure text is appropriate for processing
        if not text or not text.strip():
            return EnhancedTranslationResult(
                debug_info=["Empty input text"],
                processing_time=0.0
            )

        if len(text) > Config.MAX_TEXT_LENGTH:
            return EnhancedTranslationResult(
                debug_info=[f"Text too long (max {Config.MAX_TEXT_LENGTH} chars)"],
                processing_time=0.0
            )

        # Tokenization and initial linguistic analysis
        tokens = advanced_tokenize(text)
        word_tokens = [t for t in tokens if t.pos != "PUNCT"]  # Filter out punctuation
        syntax_analysis = self.syntactic_analyzer.analyze_sentence(word_tokens)

        # Initialize result containers
        raw_glosses = []
        unknown_words = []
        suggestions = {}
        verb_analyses = {}
        decision_logs = []

        # Process tokens with exception handling and pattern matching
        i = 0
        while i < len(word_tokens):
            token_processed = False

            # Check for exception patterns first (highest priority)
            # These handle special cases like reflexive verbs and contractions
            for pattern_tokens, target_gloss in self.exception_patterns:
                pattern_length = len(pattern_tokens)
                if i + pattern_length <= len(word_tokens):
                    # Extract window of tokens to match against pattern
                    window = [t.normalized for t in word_tokens[i:i + pattern_length]]
                    if window == pattern_tokens:
                        mapped_gloss = self._map_token(target_gloss)
                        if mapped_gloss:
                            raw_glosses.append(mapped_gloss)
                            decision_logs.append(DecisionLog(
                                step='exception_pattern',
                                input_data={'pattern': window},
                                output_data={'gloss': mapped_gloss},
                                rule_or_model='exception_manager',
                                confidence=0.95
                            ))
                            i += pattern_length  # Skip processed tokens
                            token_processed = True
                            break

            if token_processed:
                continue

            # Process individual token if no pattern matched
            current_token = word_tokens[i]
            i += 1

            # Skip reflexive clitics when standalone (they'll be handled with verbs)
            if current_token.normalized in LSFRules.REFLEXIVE_CLITICS:
                # Check if next token is a verb that should be reflexive
                if i < len(word_tokens):
                    next_token = word_tokens[i]
                    # Special handling for common reflexive verbs
                    if next_token.normalized in ['reveille', 'leve', 'couche', 'lave', 'habille']:
                        reflexive_infinitive = f"se_{next_token.normalized.replace('e', '')}"
                        if next_token.normalized.endswith('e'):
                            reflexive_infinitive = f"se_{next_token.normalized[:-1]}er"

                        mapped_gloss = self._map_token(reflexive_infinitive)
                        if mapped_gloss:
                            raw_glosses.append(mapped_gloss)
                            i += 1  # Skip the next token as we processed it
                            continue

                # Log decision to skip reflexive clitic
                decision_logs.append(DecisionLog(
                    step='skip_reflexive',
                    input_data={'token': current_token.text},
                    output_data={},
                    rule_or_model='lsf_rules',
                    confidence=0.9
                ))
                continue

            # Handle personal pronouns using LSF pronoun mapping
            if current_token.normalized in LSFRules.PRONOUN_MAPPING:
                gloss = LSFRules.PRONOUN_MAPPING[current_token.normalized]
                raw_glosses.append(gloss)
                decision_logs.append(DecisionLog(
                    step='pronoun_mapping',
                    input_data={'token': current_token.text},
                    output_data={'gloss': gloss},
                    rule_or_model='lsf_pronoun_rules',
                    confidence=0.95
                ))
                continue

            # Skip function words that are typically omitted in LSF
            # (except temporal expressions which are important)
            if (current_token.normalized in LSFRules.FUNCTION_WORDS_TO_OMIT and
                    current_token.normalized not in LSFRules.TEMPORAL_EXPRESSIONS):
                decision_logs.append(DecisionLog(
                    step='omit_function_word',
                    input_data={'token': current_token.text},
                    output_data={},
                    rule_or_model='lsf_function_word_rules',
                    confidence=0.85
                ))
                continue

            # Direct lexical mapping using dictionary lookup
            mapped_gloss = self._map_token(current_token.normalized)
            if mapped_gloss:
                raw_glosses.append(mapped_gloss)

                # Check if this is a verb for detailed analysis
                if (current_token.normalized in self.dictionary and
                        self._is_actual_verb(current_token.normalized)):
                    verb_analyses[current_token.text] = {
                        'infinitive': current_token.normalized,
                        'gloss': mapped_gloss,
                        'confidence': 0.85,
                        'is_reflexive': False
                    }

                decision_logs.append(DecisionLog(
                    step='lexical_mapping',
                    input_data={'token': current_token.text},
                    output_data={'gloss': mapped_gloss},
                    rule_or_model='dictionary_lookup',
                    confidence=0.8
                ))
                continue

            # Handle unknown words - generate suggestions and mark for attention
            unknown_words.append(current_token.text)
            suggestions[current_token.text] = self._generate_suggestions(current_token.normalized)

            # Insert unknown marker in sequence for transparency
            unknown_marker = f"{current_token.normalized}(*)"
            raw_glosses.append(unknown_marker)
            decision_logs.append(DecisionLog(
                step='unknown_word',
                input_data={'token': current_token.text},
                output_data={
                    'marker': unknown_marker,
                    'suggestions': suggestions[current_token.text]
                },
                rule_or_model='fallback_handler',
                confidence=0.1
            ))

        # Apply LSF planning if enabled - organizes glosses according to sign language grammar
        plan = None
        if use_planner:
            plan, planning_logs = self.lsf_planner.plan_sequence(
                raw_glosses,
                syntax_analysis.negation
            )
            decision_logs.extend(planning_logs)
            final_glosses = plan.gloss_sequence
        else:
            final_glosses = raw_glosses

        # Apply case formatting according to LSF convention
        if uppercase_glosses:
            final_glosses = [g.upper() for g in final_glosses]
        else:
            final_glosses = [g.lower() for g in final_glosses]

        # Comprehensive quality evaluation
        quality_metrics = self.quality_evaluator.evaluate(
            text, final_glosses, unknown_words, syntax_analysis, plan
        )

        # Compile complete result with all analysis data
        processing_time = time.time() - start_time

        return EnhancedTranslationResult(
            glosses=final_glosses,
            unknown_words=unknown_words,
            suggestions=suggestions,
            quality_metrics=quality_metrics,
            syntactic_analysis=syntax_analysis,
            verb_analysis=verb_analyses,
            processing_time=processing_time,
            decision_log=decision_logs,
            plan=plan,
            debug_info=[
                f"Processed {len(word_tokens)} tokens",
                f"Generated {len(decision_logs)} decision logs",
                f"Overall quality: {quality_metrics.overall_quality:.1%}"
            ]
        )

    def _map_token(self, token: str) -> Optional[str]:
        """
        Map token to LSF gloss using multiple strategies
        Tries various normalization and fuzzy matching approaches
        """
        normalized_token = normalize_token(token, Config.IGNORE_ACCENTS)

        # Check main dictionary first
        if normalized_token in self.dictionary:
            return self.dictionary[normalized_token]

        # Try various normalization variations
        variations = [
            normalized_token.replace(" ", "_"),
            normalized_token.replace("_", " "),
            canonicalize(normalized_token)
        ]

        for variation in variations:
            if variation in self.dictionary:
                return self.dictionary[variation]

        # Check video index as fallback
        for variation in [normalized_token] + variations:
            if variation in self.video_index:
                return variation

        return None

    def _is_actual_verb(self, word: str) -> bool:
        """
        Check if a word is actually a verb (not an adverb, noun, etc.)
        Important for proper linguistic analysis and verb conjugation handling
        """
        normalized = normalize_token(word, Config.IGNORE_ACCENTS)

        # Exclude temporal expressions (these are adverbs, not verbs)
        if normalized in LSFRules.TEMPORAL_EXPRESSIONS:
            return False

        # Exclude pronouns
        if normalized in LSFRules.PRONOUN_MAPPING:
            return False

        # Exclude function words
        if normalized in LSFRules.FUNCTION_WORDS_TO_OMIT:
            return False

        # Check against comprehensive list of common nouns (not verbs)
        common_nouns = {
            "voiture", "maison", "travail", "école", "bureau", "train", "bus",
            "ami", "amie", "famille", "enfant", "homme", "femme", "chien",
            "chat", "livre", "table", "chaise", "porte", "fenêtre", "clé",
            "argent", "eau", "pain", "lait", "café", "thé", "vin", "bière",
            "pomme", "banane", "orange", "fraise", "tomate", "carotte",
            "pomme de terre", "riz", "pâtes", "viande", "poisson", "œuf",
            "fromage", "beurre", "sucre", "sel", "poivre", "huile", "travail",
            "voiture", "maison", "école", "bureau", "train", "bus", "avion",
            "bateau", "vélo", "moto", "camion", "taxi", "métro", "tramway",
            "rue", "place", "parc", "jardin", "forêt", "montagne", "mer",
            "rivière", "lac", "plage", "ville", "pays", "monde", "terre",
            "ciel", "soleil", "lune", "étoile", "nuage", "pluie", "neige",
            "vent", "tempête", "orage", "arc-en-ciel", "jour", "nuit", "matin",
            "soir", "midi", "minuit", "semaine", "mois", "année", "saison",
            "printemps", "été", "automne", "hiver", "janvier", "février",
            "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
            "octobre", "novembre", "décembre", "lundi", "mardi", "mercredi",
            "jeudi", "vendredi", "samedi", "dimanche"
        }
        if normalized in common_nouns:
            return False

        # Check verb endings and validate against conjugation dictionary
        verb_endings = {"er", "ir", "re", "oir", "ais", "ait", "ons", "ez", "ent",
                        "ai", "as", "a", "ont", "es", "est", "sont"}

        # Only consider it a verb if it has proper verb characteristics
        if any(normalized.endswith(end) for end in verb_endings):
            if normalized in self.complete_conjugation_dict:
                # Additional check: exclude temporal expressions that might be in dict
                temporal_words = {"bientot", "bientôt", "demain", "hier", "maintenant", "aujourdhui", "aujourd'hui"}
                if normalized not in temporal_words:
                    return True

        # For words in the conjugation dictionary, apply strict validation
        if normalized in self.complete_conjugation_dict:
            temporal_words = {"bientot", "bientôt", "demain", "hier", "maintenant", "aujourdhui", "aujourd'hui"}
            if normalized not in temporal_words:
                # Must have clear verb characteristics
                verb_generator = VerbConjugationGenerator()
                irregular_verbs = set(verb_generator.irregular_verbs.keys())

                # Validation criteria:
                # 1. Ends with verb endings, OR
                # 2. Is a known irregular verb, OR
                # 3. Is a conjugated form of a known verb
                if (any(normalized.endswith(end) for end in verb_endings) or
                        normalized in irregular_verbs):
                    return True

                # Check if it's a conjugated form by prefix matching
                for verb in irregular_verbs:
                    if normalized.startswith(verb) and len(normalized) > len(verb):
                        return True

        return False

    def _generate_suggestions(self, word: str) -> List[str]:
        """
        Generate suggestions for unknown words using fuzzy matching
        Helps users understand potential alternatives for unrecognized terms
        """
        all_keys = list(self.dictionary.keys()) + list(self.video_index.keys())

        # Use Python's difflib for close string matches
        close_matches = difflib.get_close_matches(
            word, all_keys, n=3, cutoff=0.6
        )

        suggestions = close_matches[:]

        # Add morphological suggestions for longer words
        if len(word) > 3:
            for key in all_keys[:100]:  # Limit search scope for performance
                if len(key) > 3:
                    # Check prefix or suffix similarity
                    if (word[:3] == key[:3] or word[-3:] == key[-3:]) and key not in suggestions:
                        suggestions.append(key)
                        if len(suggestions) >= 5:
                            break

        return suggestions[:5]  # Limit to 5 suggestions


# ============================================================================
# STREAMLIT USER INTERFACE - CREATIVE MULTILINGUAL DESIGN
# ============================================================================

def render_custom_css():
    """
    Render custom CSS for the creative multilingual interface
    Implements the SLOWKATHON theme celebrating linguistic diversity
    """
    if not HAS_STREAMLIT:
        return

    st.markdown("""
    <style>
    /* SLOWKATHON Creative Theme - Multilingual & Cultural Design */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }

    /* Floating animation for creative elements */
    @keyframes float {
        0%, 100% { transform: translateX(-50%) translateY(0px); }
        50% { transform: translateX(-50%) translateY(-10px); }
    }

    .main-header h1 {
        font-size: 2.5rem;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #fff, #f0f8ff, #fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0;
        opacity: 0.95;
    }

    /* Quality metrics styling */
    .quality-metrics {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }

    /* Translation output styling */
    .translation-output {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 1.1em;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
        position: relative;
    }

    .translation-output::before {
        content: '🤟';
        position: absolute;
        top: -15px;
        left: 20px;
        background: white;
        padding: 5px 10px;
        border-radius: 50%;
        font-size: 1.2rem;
        border: 3px solid #667eea;
    }

    /* Verb analysis styling */
    .verb-analysis {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(67, 233, 123, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }

    /* Decision log styling */
    .decision-log {
        background: linear-gradient(135deg, #f1f3f4 0%, #e8f0fe 100%);
        border-left: 5px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Segoe UI', monospace;
        font-size: 0.9em;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }

    /* Creative multilingual badges */
    .multilingual-badge {
        display: inline-block;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: bold;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Statistics cards with creative styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        border: 2px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    /* Help section styling */
    .help-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(240, 147, 251, 0.3);
    }

    /* Main button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Checkbox styling - REMOVED BACKGROUND COLORS as requested */
    .stCheckbox > label {
        color: #333;
        font-weight: normal;
    }

    .stCheckbox > label > div[data-testid="stMarkdown"] {
        color: #333;
    }

    /* Input styling */
    .stTextArea > div > div > textarea {
        border: 2px solid #667eea;
        border-radius: 10px;
        font-size: 1.1rem;
        transition: border-color 0.3s ease;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #764ba2;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }

    /* Slowkathon theme elements */
    .snail-emoji {
        font-size: 2rem;
        animation: crawl 4s ease-in-out infinite;
    }

    @keyframes crawl {
        0%, 100% { transform: translateX(0); }
        50% { transform: translateX(20px); }
    }

    .language-diversity {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


def render_main_interface():
    """
    Render the main translation interface with creative multilingual design
    This is the primary user interaction point for the application
    """
    if not HAS_STREAMLIT:
        print("Streamlit interface not available. Running in console mode.")
        return

    render_custom_css()

    # Creative header with SLOWKATHON branding
    st.markdown("""
    <div class="main-header">
        <h1>🏆 FR → LSF Translator SLOWKATHON</h1>
        <p><span class="language-diversity">Multilingual Creativity</span> • Advanced LSF Translation • Cultural Expression</p>
        <p><strong>Circle U. Alliance — European Day of Languages — September 26, 2025</strong></p>
        <div style="margin-top: 1rem;">
            <span class="multilingual-badge">Français</span>
            <span class="multilingual-badge">English</span>
            <span class="multilingual-badge">LSF</span>
            <span class="multilingual-badge">Multilingual</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize translation systems with progress indication
    with st.spinner("🚀 Initializing linguistic systems..."):
        dictionary, all_glosses = load_video_dictionary()
        video_index = build_video_index(Config.VIDEOS_DIR)
        translator = EnhancedLSFTranslator(dictionary, video_index)
        video_generator = VideoGenerator(video_index)

    # Store systems in session state for persistence
    st.session_state['translator'] = translator
    st.session_state['video_generator'] = video_generator
    st.session_state['video_index'] = video_index

    # Creative system statistics display
    st.markdown("### 🎨 Creative Multilingual Database")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📚 Dictionary</h3>
            <h2>1,103</h2>
            <p>Lexical Entries</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎬 LSF Videos</h3>
            <h2>1,103</h2>
            <p>Cultural Clips</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Verbs</h3>
            <h2>3,784</h3>
            <p>Conjugated Forms</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Patterns</h3>
            <h2>18</h3>
            <p>Creative Exceptions</p>
        </div>
        """, unsafe_allow_html=True)

    # Main translation interface
    st.markdown("### 📝 Translation Interface")

    # Input section with options sidebar
    col_input, col_options = st.columns([3, 1])

    with col_input:
        text_input = st.text_area(
            "French text to translate:",
            key=SessionKeys.TEXT,
            height=120,
            placeholder="Example: Je me réveille tôt le matin et je vais au travail...",
            help=f"Enter your French text (max {Config.MAX_TEXT_LENGTH} characters)"
        )

        # Character counter with color coding
        char_count = len(text_input) if text_input else 0
        color = "red" if char_count > Config.MAX_TEXT_LENGTH else "green"
        st.markdown(f"<p style='color:{color}'>Characters: {char_count}/{Config.MAX_TEXT_LENGTH}</p>",
                    unsafe_allow_html=True)

    with col_options:
        st.markdown("**⚙️ Options**")
        # LSF Planner checkbox (canonical ordering)
        enable_planner = st.checkbox("LSF Planner", value=True,
                                     help="Apply canonical LSF word order")
        # Case formatting option
        lowercase_glosses = st.checkbox("Lowercase glosses", value=False,
                                        help="Output in lowercase instead of uppercase")
        # Analysis detail level
        show_analysis = st.checkbox("Detailed analysis", value=True,
                                    help="Show syntactic and verbal analysis")

    # Action buttons with clear functionality
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        translate_btn = st.button("🚀 Translate", type="primary", use_container_width=True)

    with col2:
        analyze_btn = st.button("🔍 Analyze only", use_container_width=True)

    with col3:
        if st.button("🧹 Clear all", use_container_width=True):
            # Clear all session state data
            keys_to_clear = [SessionKeys.TEXT, SessionKeys.RESULT, SessionKeys.VIDEO_BYTES, SessionKeys.VIDEO_PATH]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    with col4:
        if st.button("📖 User Manual", use_container_width=True):
            st.session_state['show_help'] = not st.session_state.get('show_help', False)

    # Process translation requests
    if translate_btn or analyze_btn:
        if not text_input or not text_input.strip():
            st.warning("⚠️ Please enter text to translate")
            return

        if len(text_input) > Config.MAX_TEXT_LENGTH:
            st.error(f"❌ Text too long (maximum {Config.MAX_TEXT_LENGTH} characters)")
            return

        # Perform translation with progress indication
        with st.spinner("🔄 Translation and analysis in progress..."):
            try:
                result = translator.translate(
                    text_input,
                    uppercase_glosses=not lowercase_glosses,
                    use_planner=enable_planner
                )

                st.session_state[SessionKeys.RESULT] = result
                st.success(f"✅ Translation completed in {result.processing_time:.3f}s")

            except Exception as e:
                st.error(f"❌ Translation error: {str(e)}")
                logger.exception("Translation error")
                return

    # Display help section if requested
    if st.session_state.get('show_help', False):
        display_help_section()

    # Display translation results if available
    if SessionKeys.RESULT in st.session_state:
        display_translation_results(
            st.session_state[SessionKeys.RESULT],
            st.session_state['video_generator'],
            show_analysis
        )


def display_help_section():
    """Display comprehensive help section with usage instructions"""
    if not HAS_STREAMLIT:
        return

    st.markdown("---")
    st.markdown("""
    <div class="help-section">
        <h2>📖 User Manual</h2>
        <p>🐌 <em>Take your time to explore, create, and express yourself!</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Overview section explaining the application purpose
    st.markdown("""
    ### 🎯 Creative Multilingual Overview
    This advanced translator converts French text into LSF (French Sign Language) gloss sequences 
    with comprehensive linguistic analysis and automatic video generation.
    """)

    # Features breakdown for user understanding
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### ✨ Main Features

        **🔤 Advanced linguistic analysis**
        - Recognition of 130+ verbs with complete conjugations
        - Detection of reflexive verbs (se réveiller, se lever...)
        - Syntactic and morphological analysis
        - Part-of-speech classification

        **🤟 Intelligent LSF translation**
        - Canonical LSF order: TIME → PLACE → SUBJECT → OBJECT → VERB
        - Handling of temporal expressions (bientôt, demain, hier...)
        - Processing of pronouns and function words
        - Automatic sequence planning
        """)

    with col2:
        st.markdown("""
        **🎬 Video generation**
        - Database of 1,103 LSF videos
        - Automatic clip concatenation
        - Export in MP4/WebM format
        - Integrated preview

        **📊 Quality metrics**
        - Lexical coverage score
        - Syntactic and morphological precision
        - LSF planning evaluation
        - Suggestions for unknown words
        """)

    # Usage instructions for clear user guidance
    st.markdown("""
    ### 🚀 How to use

    1. **Text input** : Enter your French sentence in the text area
    2. **Options** : Configure parameters according to your needs
    3. **Translation** : Click "🚀 Translate" to get the LSF sequence
    4. **Analysis** : Consult detailed verb and syntax analysis
    5. **Video** : Generate and download the corresponding LSF video
    """)

    # Practical examples for user understanding
    st.markdown("""
    ### 💡 Usage examples

    **Simple sentences:**
    - "Je mange" → `MOI MANGER`
    - "Il travaille" → `LUI TRAVAILLER`

    **Reflexive verbs:**
    - "Je me réveille" → `MOI SE_REVEILLER`
    - "Tu te lèves" → `TOI SE_LEVER`

    **With temporal expressions:**
    - "Je me réveille bientôt" → `BIENTOT MOI SE_REVEILLER`
    - "Il vient demain" → `DEMAIN LUI VENIR`

    **Complex sentences:**
    - "Je me réveille tôt et je vais au travail" → `MOI SE_REVEILLER TOT MOI ALLER TRAVAIL`
    """)

    # Technical details for transparency
    st.markdown("""
    ### 🔧 Technical details

    **Database:**
    - 1,103 lexical entries
    - 1,103 corresponding LSF videos
    - 3,784 conjugated verb forms
    - 18 exception patterns for special cases

    **Algorithms:**
    - Advanced tokenization with contraction expansion
    - LSF rule-based syntactic analyzer
    - Canonical order sequence planner
    - Conjugation generator for 130+ verbs
    """)

    # User tips for optimal usage
    st.markdown("""
    ### 💡 Usage tips

    - **Unknown words** : The system marks unrecognized words with (*) and provides suggestions
    - **LSF order** : Enable the planner to respect canonical LSF order
    - **Detailed analysis** : Check this option to see complete verb analysis
    - **Quality** : A high score indicates better translation quality
    - **Videos** : All video clips are available in optimized WebM format
    """)

    st.markdown("---")


def display_translation_results(result: EnhancedTranslationResult,
                                video_generator: VideoGenerator,
                                show_analysis: bool = True):
    """
    Display comprehensive translation results with quality metrics and analysis
    This function presents all translation data in an organized, user-friendly format
    """
    if not HAS_STREAMLIT:
        return

    # Quality metrics display - provides immediate feedback on translation quality
    metrics = result.quality_metrics
    st.markdown(f"""
    <div class="quality-metrics">
        <h3>📊 Quality Metrics</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div><strong>Overall score:</strong> {metrics.overall_quality:.1%}</div>
            <div><strong>Coverage:</strong> {metrics.confidence_score:.1%}</div>
            <div><strong>Syntactic precision:</strong> {metrics.syntactic_accuracy:.1%}</div>
            <div><strong>Morphology:</strong> {metrics.morphological_accuracy:.1%}</div>
            <div><strong>Planning:</strong> {metrics.planning_quality:.1%}</div>
            <div><strong>Translated/Total:</strong> {metrics.translated_words}/{metrics.total_words}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main translation output display
    if result.glosses:
        st.markdown("### 🤟 LSF Translation")

        # Format glosses for visual clarity
        gloss_display = " → ".join([f"`{g}`" for g in result.glosses])

        st.markdown(f"""
        <div class="translation-output">
            <h4>LSF gloss sequence:</h4>
            <div style="font-size: 1.1em; margin: 0.8rem 0; line-height: 1.5;">
                {gloss_display}
            </div>
            <div style="background: #e9ecef; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>Copy-paste:</strong><br>
                <code>{' '.join(result.glosses)}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Video generation section - only if FFmpeg is available
        if video_generator.ffmpeg_available:
            st.markdown("### 🎬 Video Generation")

            col1, col2 = st.columns([2, 1])

            with col1:
                if st.button("🎥 Generate LSF Video", type="secondary", use_container_width=True):
                    with st.spinner("🎬 Video generation in progress..."):
                        video_path = video_generator.generate_lsf_video(result.glosses)

                    if video_path and os.path.exists(video_path):
                        # Store video in session state for download and preview
                        with open(video_path, "rb") as f:
                            st.session_state[SessionKeys.VIDEO_BYTES] = f.read()
                        st.session_state[SessionKeys.VIDEO_PATH] = video_path
                        st.success("✅ Video generated successfully!")

            with col2:
                # Download button for generated video
                if SessionKeys.VIDEO_BYTES in st.session_state:
                    video_path = st.session_state.get(SessionKeys.VIDEO_PATH, "")
                    file_ext = "webm" if video_path.lower().endswith(".webm") else "mp4"
                    mime_type = f"video/{file_ext}"
                    filename = f"lsf_translation_{uuid.uuid4().hex[:8]}.{file_ext}"

                    st.download_button(
                        "📥 Download",
                        data=st.session_state[SessionKeys.VIDEO_BYTES],
                        file_name=filename,
                        mime=mime_type,
                        use_container_width=True
                    )

        # Video preview section
        if SessionKeys.VIDEO_BYTES in st.session_state:
            st.markdown("### 🎞️ Video Preview")
            video_path = st.session_state.get(SessionKeys.VIDEO_PATH, "")
            mime_type = "video/webm" if video_path.lower().endswith(".webm") else "video/mp4"
            st.video(st.session_state[SessionKeys.VIDEO_BYTES], format=mime_type)

    # Detailed linguistic analysis (if enabled by user)
    if show_analysis:
        display_detailed_analysis(result)


def display_detailed_analysis(result: EnhancedTranslationResult):
    """
    Display detailed linguistic analysis including verb analysis and decision logs
    Provides transparency into the translation process for educational purposes
    """
    if not HAS_STREAMLIT:
        return

    # Verb analysis section - shows detailed verb processing
    if result.verb_analysis:
        st.markdown(f"""
        <div class="verb-analysis">
            <h3>🔍 Verb Analysis</h3>
            <p>{len(result.verb_analysis)} verb form(s) identified</p>
        </div>
        """, unsafe_allow_html=True)

        # Display each verb with detailed information
        for verb_form, analysis in result.verb_analysis.items():
            with st.expander(f"Verb: {verb_form} → {analysis.get('gloss', 'N/A')}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Infinitive:** {analysis.get('infinitive', 'N/A')}")
                    st.write(f"**LSF Gloss:** {analysis.get('gloss', 'N/A')}")

                with col2:
                    st.write(f"**Confidence:** {analysis.get('confidence', 0):.1%}")
                    st.write(f"**Reflexive:** {'Yes' if analysis.get('is_reflexive', False) else 'No'}")

    # Unknown words section - helps users understand untranslated content
    if result.unknown_words:
        st.markdown("### ❓ Untranslated Words")
        st.write("These words have been inserted with (*) in the sequence:")

        for word in result.unknown_words:
            suggestions = result.suggestions.get(word, [])
            if suggestions:
                st.write(f"**{word}** — Suggestions: {', '.join(suggestions)}")
            else:
                st.write(f"**{word}** — No suggestions available")

    # Decision log section - provides transparency into translation decisions
    if result.decision_log:
        with st.expander(f"📋 Decision Log ({len(result.decision_log)} steps)"):
            # Show only key decisions to avoid information overload
            key_decisions = [
                log for log in result.decision_log
                if log.step in ['exception_pattern', 'lexical_mapping', 'unknown_word', 'sequence_planning']
            ]

            for i, decision in enumerate(key_decisions[:15]):  # Limit display for readability
                st.markdown(f"""
                <div class="decision-log">
                    <strong>{i + 1}. {decision.step}</strong> ({decision.rule_or_model})<br>
                    <em>Confidence: {decision.confidence:.1%}</em><br>
                    Input: {decision.input_data}<br>
                    Output: {decision.output_data}
                </div>
                """, unsafe_allow_html=True)

            if len(result.decision_log) > 15:
                st.info(f"... and {len(result.decision_log) - 15} other steps")


def initialize_session():
    """Initialize session state variables for persistent user experience"""
    if not HAS_STREAMLIT:
        return

    # Generate unique session ID for tracking
    if SessionKeys.SESSION_ID not in st.session_state:
        st.session_state[SessionKeys.SESSION_ID] = uuid.uuid4().hex

    # Initialize session data containers
    st.session_state.setdefault(SessionKeys.TRANSLATION_HISTORY, [])
    st.session_state.setdefault(SessionKeys.USER_FEEDBACK, [])


# ============================================================================
# CONSOLE MODE & TESTING FUNCTIONS
# ============================================================================

def console_mode():
    """
    Run translator in console mode for testing and development
    Provides command-line interface when Streamlit is not available
    """
    print("🏆 Enhanced LSF Translator - Console Mode")
    print("=" * 50)

    # Initialize translation systems
    print("Loading dictionary and video index...")
    dictionary, _ = load_video_dictionary()
    video_index = build_video_index(Config.VIDEOS_DIR)
    translator = EnhancedLSFTranslator(dictionary, video_index)

    print(f"✅ Initialized with {len(dictionary)} dictionary entries")
    print(f"✅ Found {len(video_index)} video files")
    print()

    # Interactive translation loop
    while True:
        try:
            text = input("📝 Entrez le texte français (ou 'quit' pour quitter): ").strip()

            if text.lower() in ['quit', 'q', 'exit']:
                print("👋 Au revoir!")
                break

            if not text:
                continue

            print("\n🔄 Traduction en cours...")
            result = translator.translate(text)

            print(f"\n🤟 Traduction LSF: {' '.join(result.glosses)}")
            print(f"⏱️  Temps de traitement: {result.processing_time:.3f}s")
            print(f"📊 Score de qualité: {result.quality_metrics.overall_quality:.1%}")

            if result.unknown_words:
                print(f"❓ Mots inconnus: {', '.join(result.unknown_words)}")

            print("-" * 50)

        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
            logger.exception("Console mode error")


def test_all_verbs():
    """
    Test comprehensive verb coverage with predefined test cases
    Validates the translation system's performance on various sentence types
    """
    print("🧪 Testing verb coverage...")

    # Initialize translator system
    dictionary, _ = load_video_dictionary()
    video_index = build_video_index(Config.VIDEOS_DIR)
    translator = EnhancedLSFTranslator(dictionary, video_index)

    # Test cases covering different grammatical constructions
    test_cases = [
        # Reflexive verbs - critical for LSF
        ("Je me réveille", ["MOI", "SE_REVEILLER"]),
        ("Tu te lèves", ["TOI", "SE_LEVER"]),
        ("Elle se couche", ["ELLE", "SE_COUCHER"]),

        # Regular verbs - basic sentence structures
        ("Je mange", ["MOI", "MANGER"]),
        ("Il travaille", ["LUI", "TRAVAILLER"]),
        ("Nous étudions", ["NOUS", "ETUDIER"]),

        # Irregular verbs - challenging conjugations
        ("Je vais au travail", ["MOI", "ALLER", "TRAVAIL"]),
        ("Elle vient demain", ["ELLE", "VENIR", "DEMAIN"]),
        ("Nous avons mangé", ["NOUS", "AVOIR", "MANGE"]),

        # Complex sentences - real-world usage
        ("Je me réveille tôt et je vais travailler",
         ["MOI", "SE_REVEILLER", "TOT", "MOI", "ALLER", "TRAVAILLER"]),
    ]

    success_count = 0

    # Run each test case and evaluate results
    for i, (french, expected) in enumerate(test_cases, 1):
        result = translator.translate(french, uppercase_glosses=True)
        actual = [g for g in result.glosses if not g.endswith('(*)')]

        # Flexible matching - check if key elements are present
        matches = sum(1 for exp in expected if any(exp.lower() in act.lower() for act in actual))
        coverage = matches / len(expected)

        if coverage >= 0.6:  # 60% coverage threshold for success
            print(f"✅ Test {i}: {french}")
            success_count += 1
        else:
            print(f"❌ Test {i}: {french}")
            print(f"   Expected: {expected}")
            print(f"   Got: {actual}")
            print(f"   Coverage: {coverage:.1%}")

    # Report overall test results
    success_rate = 100 * success_count / len(test_cases)
    print(f"\n📊 Success rate: {success_count}/{len(test_cases)} ({success_rate:.1f}%)")
    return success_count == len(test_cases)


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main application entry point - coordinates the entire application
    Handles both Streamlit web interface and console mode execution
    """
    try:
        if HAS_STREAMLIT:
            # Configure Streamlit application settings
            try:
                st.set_page_config(
                    page_title="🏆 Traducteur FR → LSF SLOWKATHON",
                    layout="wide",
                    initial_sidebar_state="expanded",
                    page_icon="🤟"
                )
            except Exception as e:
                logger.warning(f"Streamlit config warning: {e}")

            # Initialize session and render main interface
            initialize_session()
            render_main_interface()

            # Creative footer with SLOWKATHON branding
            st.markdown("---")
            st.markdown("""
            <div class="footer">
                <h3>🏆 SLOWKATHON — Creative Multilingual Translator</h3>
                <p><span class="language-diversity">Celebrating Linguistic Diversity</span> • Cultural Expression • Creative Technology</p>
                <p><strong>Developed for Circle U. Alliance — European Day of Languages 2025 by Yanis PISON</strong></p>
                <div style="margin-top: 1rem; font-size: 1.2rem;">
                    <span class="snail-emoji">🐌</span>
                    <span style="margin: 0 1rem;">•</span>
                    <span>🌍 Multilingual</span>
                    <span style="margin: 0 1rem;">•</span>
                    <span>🎭 Creative</span>
                    <span style="margin: 0 1rem;">•</span>
                    <span>🤟 Inclusive</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback to console mode when Streamlit is unavailable
            console_mode()

    except Exception as e:
        logger.exception("Critical application error")
        if HAS_STREAMLIT:
            st.error(f"❌ Critical application error: {str(e)}")
            with st.expander("🔧 Debug information"):
                st.code(traceback.format_exc())
        else:
            print(f"❌ Critical error: {e}")
            print(traceback.format_exc())


# Application entry point with command line argument support
if __name__ == "__main__":
    # Command line argument handling for different execution modes
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Run comprehensive test suite
            test_all_verbs()
        elif sys.argv[1] == "console":
            # Force console mode execution
            console_mode()
        else:
            print("Usage: python lsf_translator.py [test|console]")
    else:
        # Default: run main application (Streamlit if available, console otherwise)
        main()
