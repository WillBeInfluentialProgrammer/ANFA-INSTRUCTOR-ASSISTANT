# vector_store.py
import os
import sys
import pickle
import hashlib
import shutil
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from chromadb.config import Settings

# Force UTF-8 encoding for stdout to handle emojis on Windows
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

class VectorStoreManager:
    """Manage Chroma vector store for document retrieval with hybrid search support"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = "anf_documents"
        self.bm25_cache_path = os.path.join(persist_directory, "bm25_cache.pkl")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = None
        self.bm25 = None
        self.all_docs_bm25 = []

    def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        """Load all PDF documents from the specified folder"""
        all_docs = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"[INFO] Loaded {len(docs)} pages from {filename}")
                except Exception as e:
                    print(f"[WARNING] Error loading {filename}: {e}")
        return all_docs

    def create_vectorstore(self, documents: List[Document]):
        """Create and persist vector store from documents"""
        # Clean up any corrupted database first
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        
        os.makedirs(self.persist_directory, exist_ok=True)

        # Create Chroma client with proper settings to avoid tenant issues
        client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Chroma auto-saves to persist_directory
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            client=client
        )
        
        # Build and cache BM25 index for hybrid search
        print(f"[INFO] Building and caching BM25 index for hybrid search...")
        self._build_and_cache_bm25()
        
        print(f"[SUCCESS] Vector store created with {len(documents)} documents at {self.persist_directory}")

    def load_vectorstore(self):
        """Load existing Chroma vector store"""
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"No Chroma DB found at {self.persist_directory}")

        try:
            # Create Chroma client with proper settings
            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                client=client
            )
        except Exception as e:
            if "tenant" in str(e).lower():
                print(f"[ERROR] Database corrupted. Removing: {self.persist_directory}")
                shutil.rmtree(self.persist_directory)
                raise ValueError("Database was corrupted. Please restart and process documents again.")
            raise
        
        # Try to load BM25 from cache, rebuild if not found
        if not self._load_bm25_cache():
            print(f"[INFO] BM25 cache not found, building new index...")
            self._build_and_cache_bm25()
        
        print(f"[INFO] Existing vector store loaded from {self.persist_directory}")

    def initialize_or_load(self, data_folder: str):
        """Load existing DB or create a new one if not exists"""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("[INFO] Existing vector DB found - loading...")
            self.load_vectorstore()
        else:
            print("[INFO] No existing DB found - creating a new one...")
            docs = self.load_documents_from_folder(data_folder)
            if not docs:
                raise ValueError("[ERROR] No documents found in the data folder!")
            self.create_vectorstore(docs)

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Perform semantic similarity search"""
        if not self.vectorstore:
            raise ValueError("❌ Vector store not initialized.")
        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 5):
        """Perform similarity search with relevance scores"""
        if not self.vectorstore:
            raise ValueError("❌ Vector store not initialized.")
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def max_marginal_relevance_search(self, query: str, k: int = 5, fetch_k: int = 20):
        """Perform Maximal Marginal Relevance (MMR) search for diversity"""
        if not self.vectorstore:
             raise ValueError("[ERROR] Vector store not initialized.")
        return self.vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

    # --- Hybrid Search Implementation ---
    def _build_and_cache_bm25(self):
        """Build BM25 index and cache it to disk for faster subsequent loads"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[WARNING] rank_bm25 not installed. Hybrid search will fall back to semantic only.")
            return

        if not self.vectorstore:
            return

        # Fetch all documents to build BM25 index (in-memory)
        print("[INFO] Building BM25 index from vector store...")
        result = self.vectorstore.get() # raw get from chroma
        texts = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        ids = result.get("ids", [])
        
        # Reconstruct Document objects for internal mapping
        self.all_docs_bm25 = []
        tokenized_corpus = []
        
        for i, text in enumerate(texts):
            if text:
                # Store document with unique ID for better tracking
                metadata = metadatas[i] if i < len(metadatas) else {}
                metadata['_chroma_id'] = ids[i] if i < len(ids) else str(i)
                
                doc = Document(page_content=text, metadata=metadata)
                self.all_docs_bm25.append(doc)
                tokenized_corpus.append(text.lower().split())

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            # Cache BM25 index and documents to disk
            cache_data = {
                'bm25': self.bm25,
                'docs': self.all_docs_bm25,
                'version': self._get_db_version()
            }
            
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                with open(self.bm25_cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"[SUCCESS] BM25 index built and cached ({len(tokenized_corpus)} documents)")
            except Exception as e:
                print(f"[WARNING] Could not cache BM25 index: {e}")
        else:
            self.bm25 = None

    def _load_bm25_cache(self) -> bool:
        """Load BM25 index from cache if available and valid"""
        if not os.path.exists(self.bm25_cache_path):
            return False
        
        try:
            with open(self.bm25_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache version matches current DB
            if cache_data.get('version') != self._get_db_version():
                print("[INFO] BM25 cache outdated, will rebuild")
                return False
            
            self.bm25 = cache_data['bm25']
            self.all_docs_bm25 = cache_data['docs']
            print(f"[SUCCESS] Loaded BM25 index from cache ({len(self.all_docs_bm25)} documents)")
            return True
            
        except Exception as e:
            print(f"[WARNING] Could not load BM25 cache: {e}")
            return False

    def _get_db_version(self) -> str:
        """Generate version hash of current database for cache validation"""
        if not self.vectorstore:
            return ""
        
        try:
            result = self.vectorstore.get()
            ids = result.get("ids", [])
            # Hash the sorted IDs to create a version identifier
            version_string = "".join(sorted(ids))
            return hashlib.md5(version_string.encode()).hexdigest()[:16]
        except:
            return ""

    def _get_doc_id(self, doc: Document) -> str:
        """
        Generate unique identifier for a document based on metadata.
        Priority: 1) Chroma ID, 2) Act+Section+Chunk, 3) Content hash
        """
        # Try Chroma ID first
        if '_chroma_id' in doc.metadata:
            return doc.metadata['_chroma_id']
        
        # Try metadata-based ID
        act = doc.metadata.get('act', '')
        section = doc.metadata.get('section_number', '')
        chunk = doc.metadata.get('chunk_id', '')
        
        if act or section or chunk:
            return f"{act}|{section}|{chunk}"
        
        # Fallback to content hash
        return hashlib.md5(doc.page_content[:500].encode()).hexdigest()

    def _calculate_metadata_boost(self, doc: Document, query: str, 
                                   section_num: Optional[str] = None,
                                   act_name: Optional[str] = None) -> float:
        """
        Calculate metadata-based relevance boost for legal documents.
        Returns a boost score (1.0 = no boost, >1.0 = boosted)
        """
        boost = 1.0
        metadata = doc.metadata
        
        # Exact section match: Massive 20x boost to ensure correct section is prioritized
        if section_num:
            doc_section = str(metadata.get('section', metadata.get('section_number', '')))
            if doc_section == str(section_num):
                boost *= 20.0
            
        # Exact act match: Massive 10x boost to filter out manuals/other acts
        if act_name:
            doc_law = metadata.get('law', metadata.get('act', '')).lower()
            if act_name.lower() in doc_law:
                boost *= 10.0
            
        # Section mentioned in query and present in metadata: +50% boost
        if section_num and section_num in query.lower():
            doc_section = str(metadata.get('section', metadata.get('section_number', '')))
            if doc_section:
                boost *= 1.5
            
        return boost    

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7, filter: dict = None,
                     section_num: Optional[str] = None, act_name: Optional[str] = None) -> List[Document]:
        """
        Enhanced Hybrid Search with metadata-aware ranking.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight between semantic (1.0) and keyword (0.0) - default 0.7 (favors semantic more)
            filter: Optional metadata filter dict (e.g., {"act": "CNSA", "section": "9"})
            section_num: Optional section number for metadata boosting
            act_name: Optional act name for metadata boosting
        
        Returns:
            List of Document objects ranked by hybrid score with metadata boosting
        """
        if not self.vectorstore:
             raise ValueError("❌ Vector store not initialized.")
        
        # Determine fetch multiplier based on filtering - increased for better retrieval
        fetch_multiplier = 5 if filter else 4
        
        # 1. Semantic Search (Vector) with optional metadata filter
        try:
            if filter:
                semantic_results = self.vectorstore.similarity_search(query, k=k*fetch_multiplier, filter=filter)
            else:
                semantic_results = self.vectorstore.similarity_search(query, k=k*fetch_multiplier)
        except Exception as e:
            print(f"⚠️ Semantic search with filter failed: {e}")
            # Fallback without filter
            semantic_results = self.vectorstore.similarity_search(query, k=k*fetch_multiplier)
        
        # 2. Keyword Search (BM25)
        if not hasattr(self, 'bm25') or self.bm25 is None:
            # Lazy load BM25 if not ready
            self._build_and_cache_bm25()
            
        bm25_results = []
        if hasattr(self, 'bm25') and self.bm25:
            tokenized_query = query.lower().split()
            # Get top N BM25 matches (fetch more to allow for filtering)
            bm25_docs = self.bm25.get_top_n(tokenized_query, self.all_docs_bm25, n=k*fetch_multiplier)
            
            # Apply metadata filter to BM25 results if provided
            if filter:
                bm25_results = [
                    doc for doc in bm25_docs 
                    if all(str(doc.metadata.get(key, '')).lower() == str(value).lower() 
                          for key, value in filter.items())
                ]
            else:
                bm25_results = bm25_docs
        
        # 3. Enhanced RRF with Metadata-Aware Scoring
        k_const = 60
        doc_scores = {}
        
        # Process Semantic Results
        for rank, doc in enumerate(semantic_results):
            doc_id = self._get_doc_id(doc)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "semantic_score": 0.0, "bm25_score": 0.0}
            
            # RRF score for semantic search
            rrf_score = 1.0 / (rank + k_const)
            doc_scores[doc_id]["semantic_score"] = rrf_score
            
        # Process BM25 Results
        for rank, doc in enumerate(bm25_results):
            doc_id = self._get_doc_id(doc)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "semantic_score": 0.0, "bm25_score": 0.0}
            
            # RRF score for BM25 search
            rrf_score = 1.0 / (rank + k_const)
            doc_scores[doc_id]["bm25_score"] = rrf_score
        
        # 4. Calculate final scores with alpha weighting and metadata boost
        for doc_id, scores in doc_scores.items():
            # Weighted combination of semantic and BM25 scores
            base_score = (alpha * scores["semantic_score"]) + ((1 - alpha) * scores["bm25_score"])
            
            # Apply metadata boost for exact section/act matches
            metadata_boost = self._calculate_metadata_boost(
                scores["doc"], query, section_num, act_name
            )
            
            scores["final_score"] = base_score * metadata_boost
        
        # 5. Sort by final score and return top k
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["final_score"], reverse=True)
        

        # Return top k Document objects
        return [item["doc"] for item in sorted_docs[:k]]
