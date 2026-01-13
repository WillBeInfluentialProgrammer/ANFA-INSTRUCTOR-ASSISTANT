import sys
import os
import re
import streamlit as st
from document_processor import EnhancedDocumentProcessor
from vector_store import VectorStoreManager
from question_generator import QuestionGenerator
import json
from typing import List, Dict, Tuple, Optional
from rapidfuzz import fuzz, process
import numpy as np

# Force UTF-8 encoding for stdout/stderr to handle emojis on Windows
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Page configuration
st.set_page_config(
    page_title="ANFA Instructor Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🛡️"
)

# ANF Colors
ANF_GREEN = "#14532d"
ANF_LIGHT_GREEN = "#f6fff8"
ANF_ACCENT = "#1e7c4c"
AVATAR = "https://upload.wikimedia.org/wikipedia/en/7/70/Anti-Narcotics_Force_Logo.png"

# Custom CSS
st.markdown(f"""
<style>
    /* Main theme colors */
    :root {{
        --anf-green: {ANF_GREEN};
        --anf-light-green: {ANF_LIGHT_GREEN};
        --anf-accent: {ANF_ACCENT};
    }}
    
    /* Header styling */
    .main-header {{
        background: linear-gradient(135deg, {ANF_GREEN} 0%, {ANF_ACCENT} 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .main-title {{
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }}
    
    .sub-title {{
        color: {ANF_LIGHT_GREEN};
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }}
    
    /* Card styling */
    .feature-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid {ANF_ACCENT};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }}
    
    .feature-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    
    .feature-title {{
        color: {ANF_GREEN};
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: {ANF_GREEN};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }}
    
    .stButton>button:hover {{
        background-color: {ANF_ACCENT};
        transform: scale(1.05);
    }}
    
    /* Question output styling */
    .question-output {{
        background: {ANF_LIGHT_GREEN};
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid {ANF_ACCENT};
        margin-top: 1rem;
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: {ANF_LIGHT_GREEN};
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {ANF_LIGHT_GREEN};
        border-radius: 5px;
        color: {ANF_GREEN};
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {ANF_GREEN};
        color: white;
    }}
    
    /* Info boxes */
    .stInfo {{
        background-color: {ANF_LIGHT_GREEN};
        border-left-color: {ANF_ACCENT};
    }}
    
    /* Success boxes */
    .stSuccess {{
        background-color: {ANF_LIGHT_GREEN};
        border-left-color: {ANF_GREEN};
    }}
    
    /* Number input */
    .stNumberInput input {{
        border-color: {ANF_ACCENT};
    }}
    
    /* Text area */
    .stTextArea textarea {{
        border-color: {ANF_ACCENT};
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SMART DOCUMENT MATCHING SYSTEM - Automatically scales with new JSON files
# ============================================================================

class SmartDocumentMatcher:
    """Intelligent document matching using fuzzy + semantic matching"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.documents_map = {}
        self.embeddings_model = None
        self.load_documents_metadata()
    
    def load_documents_metadata(self):
        """Dynamically load all document metadata from JSON files"""
        if not os.path.exists(self.data_folder):
            return
        
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.json'):
                try:
                    file_path = os.path.join(self.data_folder, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        metadata = data.get('document_metadata', {})
                        
                        # Get law/rule/manual/act name (supports all document types)
                        law_name = (metadata.get('law') or 
                                   metadata.get('rule') or 
                                   metadata.get('manual') or
                                   metadata.get('act') or
                                   metadata.get('law_name', ''))
                        
                        if law_name:
                            self.documents_map[law_name] = {
                                'keywords': metadata.get('keywords', []),
                                'filename': filename,
                                'category': metadata.get('category', ''),
                                'subcategories': metadata.get('subcategories', [])
                            }
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def get_all_searchable_terms(self) -> List[Tuple[str, str]]:
        """Create list of all searchable terms mapped to law names"""
        all_terms = []
        
        for law_name, info in self.documents_map.items():
            # Add law name itself
            all_terms.append((law_name.lower(), law_name))
            
            # Add abbreviations extracted from law name
            abbreviations = self.extract_abbreviations(law_name)
            for abbr in abbreviations:
                all_terms.append((abbr.lower(), law_name))
            
            # Add all keywords
            for keyword in info['keywords']:
                all_terms.append((keyword.lower(), law_name))
            
            # Add subcategories
            for subcat in info.get('subcategories', []):
                all_terms.append((subcat.lower(), law_name))
        
        return all_terms
    
    def extract_abbreviations(self, text: str) -> List[str]:
        """Extract common abbreviations from document name"""
        abbrs = []
        
        # Extract capital letters (e.g., "Anti Narcotics Force" → "ANF")
        capitals = ''.join([c for c in text if c.isupper()])
        if len(capitals) >= 2:
            abbrs.append(capitals)
        
        # Common abbreviation patterns
        abbr_map = {
            'Control of Narcotic Substances Act': ['CNSA', 'CNS'],
            'Pakistan Penal Code': ['PPC'],
            'Code of Criminal Procedure': ['CrPC', 'CRPC'],
            'Anti-Money Laundering Act': ['AMLA', 'AML'],
            'Qanun-e-Shahadat Order': ['QSO'],
            'Rules of Business': ['ROB'],
            'Civil Servants Act': ['CSA'],
            'Efficiency and Discipline Rules': ['EDR', 'E&D'],
            'Revised Leave Rules': ['RLR'],
            'Appointment, Promotion and Transfer': ['APT'],
            'anti narcotics force act': ['ANF'],
            'Intelligence Operations Manual': ['Intel'],
            'Drug Demand Reduction': ['DDR'],
            'Enforcement Operations': ['Enforcement'],
        }
        
        for pattern, abbr_list in abbr_map.items():
            if pattern.lower() in text.lower():
                abbrs.extend(abbr_list)
        
        return abbrs
    
    def fuzzy_match(self, query: str, threshold: int = 60) -> Optional[str]:
        """Use fuzzy matching to find best document match"""
        query_lower = query.lower().strip()
        
        # Get all searchable terms
        all_terms = self.get_all_searchable_terms()
        
        if not all_terms:
            return None
        
        # Find best match using fuzzy matching
        best_match = process.extractOne(
            query_lower,
            [term[0] for term in all_terms],
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold
        )
        
        if best_match:
            matched_term = best_match[0]
            # Find which document this term belongs to
            for term, law_name in all_terms:
                if term == matched_term:
                    return law_name
        
        return None
    
    def semantic_match(self, query: str, threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """Use semantic similarity for intelligent matching"""
        if not self.documents_map:
            return None, 0.0
        
        # Lazy load embeddings model (only when needed)
        if self.embeddings_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Error loading embeddings model: {e}")
                return None, 0.0
        
        try:
            # Encode query
            query_embedding = self.embeddings_model.encode(query)
            
            best_match = None
            best_score = 0.0
            
            for law_name, info in self.documents_map.items():
                # Create rich text representation
                text_repr = f"{law_name} {' '.join(info['keywords'])} {' '.join(info.get('subcategories', []))}"
                
                # Encode document representation
                doc_embedding = self.embeddings_model.encode(text_repr)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = law_name
            
            if best_score >= threshold:
                return best_match, best_score
            
        except Exception as e:
            print(f"Error in semantic matching: {e}")
        
        return None, 0.0

# Initialize global document matcher (loads once)
@st.cache_resource(ttl=None, show_spinner=False)
def get_document_matcher(_reload=False):
    """Cache the document matcher to avoid reloading on every query
    Args:
        _reload: Internal parameter to force cache refresh (use underscore to exclude from caching key)
    """
    return SmartDocumentMatcher(data_folder="data")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'question_generator' not in st.session_state:
    st.session_state.question_generator = QuestionGenerator()
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

def initialize_system():
    """Initialize the document processing and vector store system"""
    with st.spinner("🔄 Initializing ANF Legal Document System..."):
        try:
            # Check if vector store already exists
            if os.path.exists("./chroma_db"):
                vector_store = VectorStoreManager()
                vector_store.load_vectorstore()
                st.session_state.vector_store = vector_store
                st.session_state.documents_loaded = True
                st.success("✅ System initialized successfully!")
            else:
                st.warning("⚠️ No vector store found. Please process documents first.")
                
                # Check if documents folder exists
                if not os.path.exists("./documents"):
                    os.makedirs("./documents")
                    st.info("📁 Created 'documents' folder. Please add your PDF files there and click 'Process Documents'.")
                
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")

# Auto-initialize if DB exists and not already loaded (Runs once per session)
if not st.session_state.documents_loaded and os.path.exists("./chroma_db"):
    initialize_system()

def process_documents():
    """Process all documents and create vector store"""
    if not os.path.exists(r"data"):
        st.error("❌ Documents folder not found!")
        return
    
    files = [f for f in os.listdir(r"data") if f.endswith('.pdf') or f.endswith('.json')]
    if not files:
        st.error("❌ No PDF or JSON files found in documents folder!")
        return
    
    with st.spinner(f"📚 Processing {len(files)} documents..."):
        try:
            # Process documents
            processor = EnhancedDocumentProcessor()
            all_documents, all_sections = processor.process_all_enhanced_documents(r"data")
            
            # Create vector store
            vector_store = VectorStoreManager()
            vector_store.create_vectorstore(all_documents)
            
            st.session_state.vector_store = vector_store
            st.session_state.documents_loaded = True
            
            st.success(f"✅ Successfully processed {len(files)} documents with {len(all_documents)} chunks!")
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")

def extract_section_from_query(query: str) -> str:
    """Extract section/rule/article number from user query"""
    # Patterns: "Section 9", "Rule 5", "Article 2", "u/s 9", "9(1)", "Section 9-A"
    patterns = [
        r'(?:section|sec\.?|u/s|rule|article|art\.?|regulation|reg\.?|order|clause)\s+(\d+[A-Za-z]?(?:\(\d+\))?)',
        r'\b(\d+[A-Za-z]?)\s*(?:\(|\s|$)',  # Standalone numbers
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return ""

def extract_act_from_query(query: str) -> str:
    """
    HYBRID APPROACH: Intelligent document matching with 3-tier strategy
    Tier 1: Exact abbreviation matches (fastest - common abbreviations)
    Tier 2: Fuzzy matching (handles typos, variations, partial words)
    Tier 3: Semantic matching (AI-powered, understands context)
    
    ✅ Automatically scales - reads from JSON document_metadata
    ✅ No manual mapping needed for new documents
    ✅ Handles all variations, typos, and natural language
    """
    
    if not query:
        return ""
    
    query_lower = query.lower().strip()
    query_words = query_lower.split()
    
    # ============================================================================
    # TIER 1: Fast exact abbreviation matches (only common abbreviations)
    # ============================================================================
    # These are the ONLY manual entries needed - just common abbreviations!
    common_abbreviations = {
        'anf': 'anti narcotics force act',
        'ppc': 'pakistan penal code',
        'crpc': 'code of criminal procedure',
        'cr.p.c': 'code of criminal procedure',
        'cr.p.c.': 'code of criminal procedure',
        'cnsa': 'control of narcotic substances act',
        'cns': 'control of narcotic substances act',
        'amla': 'anti-money laundering act',
        'aml': 'anti-money laundering act',
        'qso': 'qanun-e-shahadat order',
        'qanun': 'qanun-e-shahadat order',
        'apt': 'appointment, promotion and transfer rules',
        'rob': 'rules of business',
        'csa': 'civil servants act',
        'ppr': 'punjab police rules',
        'edr': 'efficiency and discipline rules',
        'rlr': 'revised leave rules',
        'ddr': 'drug demand reduction activities',
    }
    
    # Check for exact abbreviation match (case-insensitive)
    for word in query_words:
        word_lower = word.lower().strip('.,;:!?')  # Remove punctuation
        if word_lower in common_abbreviations:
            matched_act = common_abbreviations[word_lower]
            return matched_act
    
    # ============================================================================
    # TIER 2: Fuzzy Matching (handles variations, typos, partial words)
    # ============================================================================
    try:
        matcher = get_document_matcher()
        
        # Try fuzzy matching with high threshold first (exact/close matches)
        fuzzy_result = matcher.fuzzy_match(query, threshold=75)
        if fuzzy_result:
            return fuzzy_result
        
        # Try with lower threshold (more flexible)
        fuzzy_result = matcher.fuzzy_match(query, threshold=60)
        if fuzzy_result:
            return fuzzy_result
        
    except Exception as e:
        print(f"Fuzzy matching error: {e}")
    
    # ============================================================================
    # TIER 3: Semantic Matching (AI-powered context understanding)
    # ============================================================================
    try:
        matcher = get_document_matcher()
        semantic_result, score = matcher.semantic_match(query, threshold=0.5)
        
        if semantic_result and score > 0.6:  # High confidence
            return semantic_result
        elif semantic_result and score > 0.5:  # Medium confidence
            # Use semantic match but lower priority
            return semantic_result
            
    except Exception as e:
        print(f"Semantic matching error: {e}")
    
    return None  # No match found

def expand_comparative_query(query: str) -> list:
    """Detect and expand comparative queries into multiple sub-queries"""
    query_lower = query.lower()
    
    # Detect comparative patterns
    comparative_keywords = ['common features', 'similarities', 'differences', 'compare', 'comparison', 
                           'versus', 'vs', 'similar', 'different', 'relate', 'relationship between',
                           'distinguish', 'contrast', 'differentiate']
    
    is_comparative = any(keyword in query_lower for keyword in comparative_keywords)
    
    if not is_comparative:
        return [query]  # Not comparative, return original query
    
    print(f"🔄 DETECTED COMPARATIVE QUERY - Expanding...")
    
    # Extract key concepts by splitting on common conjunctions
    concepts = []
    
    # Try to split on conjunctions
    split_patterns = [r'\band\b', r'\bor\b', r'\bversus\b', r'\bvs\.?\b', r',']
    parts = re.split('|'.join(split_patterns), query, flags=re.IGNORECASE)
    
    for part in parts:
        part = part.strip()
        # Remove comparative words to get core concept
        cleaned = re.sub(r'\b(common features?|similarities?|differences?|compare|comparison|between|of|what are|the|with)\s*', 
                        '', part, flags=re.IGNORECASE)
        cleaned = cleaned.strip('?,. ')
        if cleaned and len(cleaned) > 3:
            concepts.append(cleaned)
    
    if len(concepts) >= 2:
        # Create individual queries for each concept WITH explicit instructions
        expanded_queries = []
        for concept in concepts:
            # Keep section numbers if present
            if re.search(r'\bsection\s+\d+', concept, re.IGNORECASE):
                expanded_queries.append(f"{concept} - provide complete definition, purpose, provisions, penalties, and procedures")
            elif re.search(r'\d+', concept):  # Has numbers (might be section)
                expanded_queries.append(f"{concept} - explain in detail with all provisions, definitions, and legal consequences")
            else:
                expanded_queries.append(f"{concept} - provide comprehensive information including definition, legal framework, procedures, and penalties if applicable")
        
        print(f"   ✅ Expanded into {len(expanded_queries)} detailed sub-queries:")
        for i, eq in enumerate(expanded_queries, 1):
            print(f"      {i}. {eq}")
        return expanded_queries
    
    return [query]  # Fallback to original if expansion failed

def analyze_query_complexity(query: str) -> Dict[str, any]:
    """Analyze query complexity to determine optimal retrieval strategy"""
    analysis = {
        'is_complex': False,
        'is_comparative': False,
        'is_multi_section': False,
        'has_multiple_acts': False,
        'recommended_k': 20,  # Default
        'query_type': 'simple'
    }
    
    query_lower = query.lower()
    
    # Check for comparative query
    comparative_keywords = ['compare', 'comparison', 'versus', 'vs', 'similarities', 'differences', 
                           'common features', 'distinguish', 'contrast']
    if any(kw in query_lower for kw in comparative_keywords):
        analysis['is_comparative'] = True
        analysis['is_complex'] = True
        analysis['recommended_k'] = 25
        analysis['query_type'] = 'comparative'
    
    # Check for multiple sections
    section_count = len(re.findall(r'section\s+\d+', query_lower))
    if section_count > 1:
        analysis['is_multi_section'] = True
        analysis['is_complex'] = True
        analysis['recommended_k'] = max(analysis['recommended_k'], 30)
        analysis['query_type'] = 'multi-section'
    
    # Check for multiple acts/laws
    act_indicators = ['amla', 'cnsa', 'anf', 'ppc', 'crpc', 'qso']
    acts_mentioned = sum(1 for act in act_indicators if act in query_lower)
    if acts_mentioned > 1:
        analysis['has_multiple_acts'] = True
        analysis['is_complex'] = True
        analysis['recommended_k'] = max(analysis['recommended_k'], 30)
        analysis['query_type'] = 'multi-act'
    
    # Check for complex legal concepts
    complex_terms = ['procedure', 'process', 'definition', 'penalty', 'punishment', 
                    'investigation', 'prosecution', 'enforcement', 'jurisdiction']
    complex_count = sum(1 for term in complex_terms if term in query_lower)
    if complex_count >= 2:
        analysis['is_complex'] = True
        analysis['recommended_k'] = max(analysis['recommended_k'], 25)
    
    # Check query length (words)
    word_count = len(query.split())
    if word_count > 15:
        analysis['is_complex'] = True
        analysis['recommended_k'] = max(analysis['recommended_k'], 25)
    
    return analysis

def normalize_act_name(act_name: str) -> str:
    """Normalize act name for fuzzy matching - remove years, punctuation, normalize whitespace"""
    if not act_name:
        return ""
    
    # Convert to lowercase
    normalized = act_name.lower()
    
    # Remove years (1860, 1997, etc.)
    normalized = re.sub(r'\b\d{4}\b', '', normalized)
    
    # Remove common punctuation but keep hyphens in names like "anti-money"
    normalized = re.sub(r'[,\.;:]', '', normalized)
    
    # Normalize whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized.strip()

def get_relevant_context(query: str, k: int = 20, return_documents: bool = False):
    """Retrieve relevant context using HYBRID SEARCH (Semantic + Keyword) with metadata-first approach
    
    Args:
        query: User query string
        k: Number of documents to retrieve (increased default to 20 for better coverage)
        return_documents: If True, returns (context, documents) tuple; if False, returns context only
    
    Returns:
        If return_documents=False: context string
        If return_documents=True: (context string, list of Document objects)
    """
    if st.session_state.vector_store is None:
        return ("", []) if return_documents else ""
    
    print(f"\n{'='*60}")
    print(f"🔍 SMART SEARCH: '{query}'")
    print(f"{'='*60}")
    
    # Analyze query complexity and adjust k value if needed
    complexity = analyze_query_complexity(query)
    if complexity['is_complex']:
        print(f"🧠 COMPLEX QUERY DETECTED: {complexity['query_type']}")
        print(f"   Recommended k: {complexity['recommended_k']}")
        k = max(k, complexity['recommended_k'])  # Use higher k for complex queries
    
    # Step 0: QUERY EXPANSION for comparative questions
    expanded_queries = expand_comparative_query(query)
    all_results = []
    
    # Process each sub-query
    for idx, sub_query in enumerate(expanded_queries, 1):
        if len(expanded_queries) > 1:
            print(f"\n📍 Processing sub-query {idx}/{len(expanded_queries)}: '{sub_query}'")
        
        # Step 1: Parse query for section and act
        query_lower = sub_query.lower().strip()
        section_num = extract_section_from_query(sub_query)
        act_name_raw = extract_act_from_query(sub_query)
        
        # Normalize act name for matching (remove extra punctuation, years, etc.)
        act_name = normalize_act_name(act_name_raw) if act_name_raw else ""
        
        print(f"🔍 Query Analysis:")
        print(f"   - Extracted Section: {section_num}")
        print(f"   - Extracted Act (Raw): {act_name_raw}")
        print(f"   - Extracted Act (Normalized): {act_name}")
        print(f"   - Original Sub-query: '{sub_query}'")
        
        # Validate query specificity for section requests  
        if section_num and not act_name:
            # Check if query contains vague terms that need clarification
            vague_terms = ['act', 'law', 'section', 'rule', 'ordinance']
            query_words = sub_query.lower().split()
            
            if any(vague_term in query_words for vague_term in vague_terms):
                # Get available acts from vector store
                try:
                    sample_docs = st.session_state.vector_store.similarity_search("", k=20)
                    available_acts = set()
                    for doc in sample_docs:
                        law_name = doc.metadata.get('law', '')
                        if law_name:
                            # Extract short names for display
                            if 'anti-narcotics force' in law_name.lower() or 'anf' in law_name.lower():
                                available_acts.add("ANF Act")
                            elif 'anti-money laundering' in law_name.lower() or 'amla' in law_name.lower():
                                available_acts.add("AMLA")
                            elif 'control of narcotic' in law_name.lower() or 'cnsa' in law_name.lower():
                                available_acts.add("CNSA")
                            elif 'civil servants' in law_name.lower():
                                available_acts.add("Civil Servants Act")
                            else:
                                # For other acts, use first 3 words
                                short_name = " ".join(law_name.split()[:3])
                                available_acts.add(short_name)
                    
                    if available_acts:
                        acts_list = "\\n".join([f"• 'Section {section_num} of **{act}**'" for act in sorted(available_acts)])
                        st.warning(f"🔍 **Please be more specific!** Found Section {section_num} in multiple acts. Try:")
                        st.info(acts_list)
                    else:
                        st.warning("🔍 **Please specify which act.** Try: 'Section X of [Act Name]'")
                        
                except Exception as e:
                    st.warning("🔍 **Please be more specific!** Try: 'Section 4 of ANF Act' or 'Section 4 of AMLA'")
                
                return ("", []) if return_documents else ""
        
        # Enhanced logging
        print(f"\n{'='*60}")
        print(f"🔍 QUERY ANALYSIS:")
        print(f"   Original Query: {sub_query}")
        print(f"   Normalized Query: {query_lower}")
        print(f"   Extracted Section: {section_num if section_num else 'None'}")
        print(f"   Extracted Act: {act_name if act_name else 'None'}")
        print(f"{'='*60}\n")
        
        results = []
        
        # Step 2: METADATA-FIRST HYBRID SEARCH (if section requested)
        if section_num:
            print(f"🎯 Attempting section-specific retrieval for Section {section_num}")
            try:
                # Use similarity search without filter, then manual filtering with normalized matching
                search_query = f"Section {section_num} {act_name if act_name else ''}"
                print(f"🔍 Search query: '{search_query}'")
                
                section_docs = st.session_state.vector_store.similarity_search(
                    search_query,
                    k=min(100, k*10)  # Get more docs for filtering
                )
                
                print(f"   Retrieved {len(section_docs)} candidate documents")
                
                # Manual filtering with normalized act name matching
                filtered_docs = []
                for doc in section_docs:
                    doc_section = str(doc.metadata.get('section', '')).strip()
                    doc_law = doc.metadata.get('law', '')
                    
                    # Check section match (exact, case-insensitive)
                    section_match = doc_section.lower() == section_num.lower()
                    
                    # Check act match (normalized fuzzy matching)
                    act_match = True
                    if act_name:
                        doc_law_normalized = normalize_act_name(doc_law)
                        act_name_normalized = normalize_act_name(act_name)
                        # Fuzzy match - check if normalized terms overlap
                        act_match = (act_name_normalized in doc_law_normalized or 
                                   doc_law_normalized in act_name_normalized)
                    
                    if section_match and act_match:
                        filtered_docs.append(doc)
                
                if filtered_docs:
                    results.extend(filtered_docs[:k])
                    print(f"✅ Found {len(filtered_docs)} matching docs (Section {section_num})")
                    if filtered_docs:
                        print(f"   Sample: {filtered_docs[0].metadata.get('law')} - Section {filtered_docs[0].metadata.get('section')}")
                else:
                    print(f"⚠️ No exact matches. Trying section-only search...")
                    # Fallback: Just match section, ignore act
                    section_only_docs = [
                        doc for doc in section_docs 
                        if str(doc.metadata.get('section', '')).strip().lower() == section_num.lower()
                    ]
                    if section_only_docs:
                        results.extend(section_only_docs[:k])
                        print(f"✅ Fallback: Found {len(section_only_docs)} docs for section {section_num}")
                        # Show which acts contain this section
                        if act_name:
                            available_acts = list(set(doc.metadata.get('law', '') for doc in section_only_docs if doc.metadata.get('law')))
                            if available_acts:
                                st.info(f"📚 Section {section_num} found in: {', '.join(available_acts[:3])}")
                                
            except Exception as e:
                print(f"⚠️ Section retrieval failed: {e}")
                import traceback
                print(traceback.format_exc())
        
        # Step 3: HYBRID SEARCH for context (conditional based on section specificity)
        # If user asked for specific section and we found it, return ONLY that section
        if section_num and results:
            print(f"✅ Found {len(results)} section-specific results. Returning only section {section_num} content.")
            all_results.extend(results[:k])
            continue  # Skip to next sub-query
        
        # Otherwise, get additional context
        remaining_k = max(k - len(results), 5)  # Always get at least 5 more results
        
        try:
            # Use HYBRID search (Semantic + BM25) for better legal term matching
            print(f"🔎 Running HYBRID search for {remaining_k} more results")
            
            # Don't use ChromaDB filter - manual filtering is more reliable
            hybrid_docs = st.session_state.vector_store.hybrid_search(
                query=sub_query,
                k=remaining_k * 3,  # Get more to allow filtering
                filter=None,
                section_num=section_num,
                act_name=act_name
            )
            
            # Manual filtering with normalized act matching
            if act_name:
                act_name_normalized = normalize_act_name(act_name)
                filtered_hybrid = []
                for doc in hybrid_docs:
                    doc_law = doc.metadata.get("law", "")
                    doc_law_normalized = normalize_act_name(doc_law)
                    # Fuzzy match using normalized names
                    if act_name_normalized in doc_law_normalized or doc_law_normalized in act_name_normalized:
                        filtered_hybrid.append(doc)
                hybrid_docs = filtered_hybrid
                print(f"📋 Filtered to {len(hybrid_docs)} docs matching act (normalized)")
            
            results.extend(hybrid_docs[:remaining_k])
        except Exception as e:
            print(f"⚠️ Hybrid search failed, falling back to similarity: {e}")
            # Fallback to plain similarity search
            try:
                semantic_docs = st.session_state.vector_store.similarity_search(
                    sub_query, 
                    k=remaining_k * 2
                )
                
                # Manual filtering with normalized matching
                if act_name:
                    act_name_normalized = normalize_act_name(act_name)
                    semantic_docs = [
                        doc for doc in semantic_docs 
                        if act_name_normalized in normalize_act_name(doc.metadata.get("law", ""))
                    ]
                
                results.extend(semantic_docs[:remaining_k])
            except Exception as e2:
                print(f"⚠️ Fallback similarity search also failed: {e2}")
        
        # Add results from this sub-query to all_results
        all_results.extend(results)
        
        if len(expanded_queries) > 1:
            print(f"   ✅ Retrieved {len(results)} documents for sub-query {idx}")
    
    # Step 4: DEDUPLICATE using metadata (NOT text content)
    seen_keys = set()
    unique_results = []
    
    for doc in all_results:
        # Create unique key from metadata (act + section + chunk)
        key = (
            doc.metadata.get("law", ""),
            doc.metadata.get("section", ""),
            doc.metadata.get("chunk_id", "")
        )
        
        if key not in seen_keys:
            unique_results.append(doc)
            seen_keys.add(key)
    
    print(f"\n📊 FINAL RETRIEVAL SUMMARY:")
    print(f"   Total Unique Documents: {len(unique_results)}")
    print(f"   Returning Top {k} documents")
    if unique_results:
        print(f"   Sample Metadata: {unique_results[0].metadata if unique_results else 'None'}")
    print(f"{'='*60}\n")
    
    # Step 5: Build context and return
    context = "\n\n".join([doc.page_content for doc in unique_results[:k]])
    
    if not context.strip():
        print("❌ No relevant context found!")
    
    if return_documents:
        return context, unique_results[:k]
    return context

# Main Header
st.markdown(f"""
<div class="main-header">
    <h1 class="main-title">🛡️ ANFA Instructor Assistant</h1>
    <p class="sub-title">Pakistan Anti-Narcotics Force Legal Document Intelligence System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for system controls
with st.sidebar:
    st.image(AVATAR, width=150)
    st.title("System Controls")
    
    if st.button("🚀 Initialize System", use_container_width=True):
        initialize_system()
    
    if st.button("📚 Process Documents", use_container_width=True):
        process_documents()
    
    if st.button("🔄 Clear Cache & Reload", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    
    st.divider()
    
    if st.session_state.documents_loaded:
        st.success("✅ System Ready")
        # Show document count
        try:
            matcher = get_document_matcher()
            doc_count = len(matcher.documents_map)
            st.info(f"📊 {doc_count} documents loaded")
        except:
            pass
    else:
        st.warning("⚠️ System Not Initialized")
    
    st.divider()
    
    st.markdown("### 📖 Available Documents")
    if os.path.exists("./documents"):
        pdf_files = [f for f in os.listdir("./documents") if f.endswith('.pdf')]
        for pdf in pdf_files:
            st.markdown(f"- {pdf}")

# Main Content Area
if not st.session_state.documents_loaded:
    st.info("ℹ️ Please initialize the system using the sidebar controls before proceeding.")
else:
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Generate MCQs",
        "📋 Generate Descriptive Questions",
        "🎯 Create Scenario Assignment",
        "💬 Q&A Assistant"
    ])
    
    # Tab 1: MCQ Generation
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        #st.markdown('<p class="feature-title">📝 Multiple Choice Questions Generator</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            mcq_topic = st.text_input(
                "Enter topic or legal provision:",
                placeholder="e.g., Offences under Anti-Narcotics Force Act",
                key="mcq_topic"
            )
        
        with col2:
            num_mcqs = st.number_input(
                "Number of MCQs:",
                min_value=1,
                max_value=20,
                value=5,
                key="num_mcqs"
            )
        
        if st.button("🎲 Generate MCQs", key="gen_mcqs"):
            if mcq_topic:
                with st.spinner("Generating MCQs..."):
                    context, documents = get_relevant_context(mcq_topic, k=7, return_documents=True)
                    if context and documents:
                        # Build JSON structure from documents for QuestionGenerator
                        structured_data = {
                            "sections": [
                                {
                                    "section": doc.metadata.get('section', ''),
                                    "title": doc.metadata.get('title', ''),
                                    "body": doc.page_content
                                }
                                for doc in documents if doc.metadata.get('section')
                            ]
                        }
                        
                        # Generate MCQs with the structured data
                        mcqs = st.session_state.question_generator.generate_mcqs(structured_data, num_mcqs)

                        st.markdown('<div class="question-output">', unsafe_allow_html=True)
                        st.text(mcqs)  # ✅ Now renders **Correct** as bold
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download MCQs",
                            data=mcqs,
                            file_name=f"ANF_MCQs_{mcq_topic[:30]}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("❌ No relevant context found for the topic!")
            else:
                st.warning("⚠️ Please enter a topic!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Descriptive Questions
    with tab2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        #st.markdown('<p class="feature-title">📋 Descriptive Questions Generator</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            desc_topic = st.text_input(
                "Enter topic or legal provision:",
                placeholder="e.g., Search and seizure procedures",
                key="desc_topic"
            )
        
        with col2:
            num_desc = st.number_input(
                "Number of Questions:",
                min_value=1,
                max_value=15,
                value=5,
                key="num_desc"
            )
        
        if st.button("📄 Generate Descriptive Questions", key="gen_desc"):
            if desc_topic:
                with st.spinner("Generating descriptive questions..."):
                    context, documents = get_relevant_context(desc_topic, k=12, return_documents=True)
                    
                    if context and documents:
                        # Build JSON structure from documents for QuestionGenerator
                        structured_data = {
                            "sections": [
                                {
                                    "section": doc.metadata.get('section', ''),
                                    "title": doc.metadata.get('title', ''),
                                    "body": doc.page_content
                                }
                                for doc in documents if doc.metadata.get('section')
                            ]
                        }
                        
                        # Generate descriptive questions with the structured data
                        questions = st.session_state.question_generator.generate_descriptive(
                            structured_data, num_desc
                        )
                        
                        st.markdown('<div class="question-output">', unsafe_allow_html=True)
                        st.markdown(questions)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Questions",
                            data=questions,
                            file_name=f"ANF_Descriptive_{desc_topic[:30]}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("❌ No relevant context found for the topic!")
            else:
                st.warning("⚠️ Please enter a topic!")
        
        st.markdown('</div>', unsafe_allow_html=True)

    
    # Tab 3: Scenario Assignment
    with tab3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        #st.markdown('<p class="feature-title">🎯 Scenario-Based Assignment Creator</p>', unsafe_allow_html=True)
        
        scenario_topic = st.text_input(
            "Enter legal area for scenario:",
            placeholder="e.g., Drug trafficking, Money laundering, Controlled substances",
            key="scenario_topic"
        )
        
        if st.button("🎬 Create Scenario Assignment", key="gen_scenario"):
            if scenario_topic:
                with st.spinner("Creating scenario-based assignment..."):
                    context, documents = get_relevant_context(scenario_topic, k=15, return_documents=True)
                    if context and documents:
                        # Build JSON structure from documents
                        structured_data = {
                            "sections": [
                                {
                                    "section": doc.metadata.get('section', ''),
                                    "title": doc.metadata.get('title', ''),
                                    "body": doc.page_content
                                }
                                for doc in documents if doc.metadata.get('section')
                            ]
                        }
                        scenario = st.session_state.question_generator.generate_scenario_assignment(structured_data)
                        
                        st.markdown('<div class="question-output">', unsafe_allow_html=True)
                        st.markdown(scenario)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Assignment",
                            data=scenario,
                            file_name=f"ANF_Scenario_{scenario_topic[:30]}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("❌ No relevant context found for the topic!")
            else:
                st.warning("⚠️ Please enter a legal area!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Q&A Assistant
    with tab4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        #st.markdown('<p class="feature-title">💬 Legal Q&A Assistant</p>', unsafe_allow_html=True)
        
        #st.markdown("Ask any question about the legal documents:")
        
        user_question = st.text_area(
            "Your Question:",
            placeholder="e.g., What are the penalties for drug trafficking under Section 9?",
            height=100,
            key="user_question"
        )
        
        if st.button("🔍 Get Answer", key="get_answer"):
            if user_question:
                with st.spinner("Searching legal documents and generating answer..."):
                    # ✅ Get both context and documents for accurate source display
                    context, source_docs = get_relevant_context(user_question, k=10, return_documents=True)  # Increased for complex queries
                    if context and source_docs:
                        # Build JSON structure from documents
                        structured_data = {
                            "sections": [
                                {
                                    "section": doc.metadata.get('section', ''),
                                    "title": doc.metadata.get('title', ''),
                                    "body": doc.page_content
                                }
                                for doc in source_docs if doc.metadata.get('section')
                            ]
                        }
                        answer = st.session_state.question_generator.answer_question(structured_data, user_question)
                        
                        st.markdown('<div class="question-output">', unsafe_allow_html=True)
                        st.markdown("### Answer:")
                        st.markdown(answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # ✅ Show source references using ACTUAL retrieved documents
                        with st.expander("📚 View Source References"):
                            if source_docs:
                                # Display top 2 sources from the actual documents used to generate the answer
                                for i, doc in enumerate(source_docs[:2], 1):
                                    st.markdown(f"**Source {i}**")
                                    
                                    # Extract document name from source field or filename
                                    doc_source = doc.metadata.get('source', 'Unknown')
                                    if '\\' in doc_source or '/' in doc_source:
                                        doc_name = doc_source.split('\\')[-1].split('/')[-1]
                                    else:
                                        doc_name = doc_source
                                    
                                    st.markdown(f"**Document:** {doc_name}")
                                    st.markdown(f"**Law:** {doc.metadata.get('law', 'Not specified')}")
                                    
                                    # Display section and title if available
                                    section = doc.metadata.get('section', 'General')
                                    title = doc.metadata.get('title', '')
                                    if title:
                                        st.markdown(f"**Section:** {section} - *{title}*")
                                    else:
                                        st.markdown(f"**Section:** {section}")
                                    
                                    # Show preview of content (remove redundant section title if present)
                                    content = doc.page_content
                                    
                                    # Remove the section title line from content preview since we already show it above
                                    if title and content.startswith(f"Section {section}: {title}"):
                                        # Skip the title line and any blank lines after it
                                        content_lines = content.split('\n')
                                        start_idx = 0
                                        for idx, line in enumerate(content_lines):
                                            if line.strip() and not line.startswith(f"Section {section}:") and not line.strip() == doc.metadata.get('law', ''):
                                                start_idx = idx
                                                break
                                        content = '\n'.join(content_lines[start_idx:])
                                    
                                    # Show preview (first 250 chars of actual content)
                                    preview = content.strip()[:250]
                                    st.text(preview + ("..." if len(content) > 250 else ""))
                                    st.divider()
                            else:
                                st.info("No source documents available.")
                    else:
                        st.error("❌ No relevant context found for your question!")
            else:
                st.warning("⚠️ Please enter a question!")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: {ANF_GREEN}; padding: 1rem;">
        <p style="margin: 0;">🛡️ <strong>Pakistan Anti-Narcotics Force</strong></p>
        <p style="margin: 0; font-size: 0.9rem;">Legal Document Intelligence System | Powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True
)
















