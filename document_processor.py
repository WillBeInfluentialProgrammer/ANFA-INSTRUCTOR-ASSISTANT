# document_processor.py 
"""
Enhanced processor that works with your existing JSON structure + new document_metadata
"""

import os
import re
import json
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class EnhancedDocumentProcessor:
    """Enhanced processor that uses document_metadata while preserving your existing structure"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def _filter_complex_metadata(self, metadata: Dict) -> Dict:
        """Filter complex metadata types to simple ones that ChromaDB can handle"""
        filtered_metadata = {}
        
        for key, value in metadata.items():
            if value is None:
                filtered_metadata[key] = None
            elif isinstance(value, (str, int, float, bool)):
                filtered_metadata[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if value:  # Only if list is not empty
                    filtered_metadata[key] = ", ".join(str(item) for item in value if item)
                else:
                    filtered_metadata[key] = ""
            elif isinstance(value, dict):
                # Convert dicts to JSON strings
                filtered_metadata[key] = json.dumps(value)
            else:
                # Convert other types to strings
                filtered_metadata[key] = str(value)
                
        return filtered_metadata

    def process_json_with_metadata(self, file_path: str) -> Tuple[List[Document], List[Dict]]:
        """Process JSON file that has document_metadata + your existing structure"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract document metadata (if present)
            doc_metadata = data.get("document_metadata", {})
            
            documents = []
            structured_sections = []
            doc_name = os.path.basename(file_path)
            
            print(f"📄 Processing: {doc_name}")
            if doc_metadata:
                print(f"✅ Found metadata: {doc_metadata.get('law_name', 'Unknown')}")
                print(f"📂 Category: {doc_metadata.get('category', 'Unknown')}")
            
            # Handle your existing structures
            if 'volumes' in data:
                # AMLA format: data['volumes']['Volume 1'][...]
                self._process_amla_format(data, doc_metadata, documents, structured_sections, doc_name)
                
            elif 'sections' in data and isinstance(data['sections'], list):
                # ANF/CNSA format: data['sections'][...]
                self._process_sections_format(data['sections'], doc_metadata, documents, structured_sections, doc_name)
                
            elif 'rules' in data and isinstance(data['rules'], list):
                # Rules format: data['rules'][...] (APT_RULES, EnD_rules_2020)
                self._process_rules_format(data['rules'], doc_metadata, documents, structured_sections, doc_name)
                
            elif isinstance(data, list):
                # Direct list format
                self._process_sections_format(data, doc_metadata, documents, structured_sections, doc_name)
                
            else:
                print(f"⚠️ Unknown format in {doc_name}")
            
            print(f"✅ Created {len(documents)} document chunks")
            return documents, structured_sections
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            return [], []

    def _process_amla_format(self, data: Dict, doc_metadata: Dict, documents: List[Document], 
                           structured_sections: List[Dict], doc_name: str):
        """Process AMLA format with volumes"""
        volumes = data.get('volumes', {})
        
        for volume_key, sections in volumes.items():
            if isinstance(sections, list):
                for section in sections:
                    self._process_section_entry(section, doc_metadata, documents, 
                                              structured_sections, doc_name, volume_key)

    def _process_sections_format(self, sections: List[Dict], doc_metadata: Dict, 
                               documents: List[Document], structured_sections: List[Dict], doc_name: str):
        """Process ANF/CNSA format with direct sections array"""
        for section in sections:
            self._process_section_entry(section, doc_metadata, documents, 
                                      structured_sections, doc_name)

    def _process_rules_format(self, rules: List[Dict], doc_metadata: Dict, 
                            documents: List[Document], structured_sections: List[Dict], doc_name: str):
        """Process rules format (APT_RULES.json, EnD_rules_2020.json) with 'rules' array"""
        for rule in rules:
            # For rules, treat 'rule' field as 'section' equivalent
            rule_entry = {
                'section': rule.get('rule', rule.get('section', '')),
                'title': rule.get('title', ''),
                'body': rule.get('body', rule.get('content', rule.get('text', ''))),
                'chapter': rule.get('chapter', '')
            }
            # Copy any additional fields
            for key, value in rule.items():
                if key not in ['rule', 'section', 'title', 'body', 'content', 'text', 'chapter']:
                    rule_entry[key] = value
            
            self._process_section_entry(rule_entry, doc_metadata, documents, 
                                      structured_sections, doc_name)

    def _process_section_entry(self, section: Dict, doc_metadata: Dict, documents: List[Document],
                             structured_sections: List[Dict], doc_name: str, volume: str = None):
        """Process individual section with enhanced metadata"""
        
        # Extract section data
        section_num = str(section.get('section', ''))
        title = section.get('title', '')
        content = section.get('body', section.get('content', ''))
        
        if not content:
            return
        
        # Build rich text with context
        law_name = doc_metadata.get('law_name', section.get('law', ''))
        
        full_text = f"{law_name}\n"
        if volume:
            full_text += f"{volume}\n"
        if section_num and title:
            full_text += f"Section {section_num}: {title}\n\n"
        full_text += content
        
        # Create enhanced metadata by combining document + section metadata
        enhanced_metadata = {
            # Document-level metadata (from document_metadata)
            "law": doc_metadata.get("law", section.get('law', '')),
            "act": doc_metadata.get("act", section.get('act', '')),
            "category": doc_metadata.get("category", ""),
            "subcategories": doc_metadata.get("subcategories", []),
            "keywords": doc_metadata.get("keywords", []),
            "jurisdiction": doc_metadata.get("jurisdiction", ""),
            "related_acts": doc_metadata.get("related_acts", []),
            
            # Section-level metadata  
            "section": section_num,
            "title": title,
            "volume": volume,
            "chapter": section.get('chapter'),
            
            # Technical metadata
            "source": doc_name,
            "document_type": "legal_act",
            "content_type": "section"
        }
        
        # Add any other fields from the original section
        for key, value in section.items():
            if key not in ['law', 'act', 'section', 'title', 'body', 'content'] and value:
                enhanced_metadata[f"original_{key}"] = value
        
        # Detect penalty information
        penalty_keywords = ['punishment', 'penalty', 'fine', 'imprisonment', 'jail', 'rigorous imprisonment']
        if any(keyword in content.lower() for keyword in penalty_keywords):
            enhanced_metadata['has_penalty'] = True
            
            # Try to extract penalty details
            penalty_info = self._extract_penalty_info(content)
            if penalty_info:
                enhanced_metadata.update(penalty_info)
        
        # Extract section-specific keywords
        section_keywords = self._extract_keywords_from_text(title + ' ' + content)
        enhanced_metadata['section_keywords'] = section_keywords
        
        # Combine all keywords for better search
        all_keywords = (doc_metadata.get("keywords", []) + 
                       section_keywords + 
                       enhanced_metadata.get("subcategories", []))
        enhanced_metadata['all_keywords'] = list(set(all_keywords))
        
        # Create document chunks
        chunks = self.text_splitter.split_text(full_text)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = enhanced_metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
            
            # Filter complex metadata for ChromaDB compatibility
            filtered_metadata = self._filter_complex_metadata(chunk_metadata)
            
            doc = Document(
                page_content=chunk,
                metadata=filtered_metadata
            )
            documents.append(doc)

    def _extract_penalty_info(self, content: str) -> Dict:
        """Extract structured penalty information from content"""
        penalty_info = {}
        content_lower = content.lower()
        
        # Extract imprisonment terms
        imprisonment_patterns = [
            r'imprisonment.*?(\d+.*?years?)',
            r'rigorous imprisonment.*?(\d+.*?years?)', 
            r'jail.*?(\d+.*?years?)'
        ]
        
        for pattern in imprisonment_patterns:
            match = re.search(pattern, content_lower)
            if match:
                penalty_info['imprisonment'] = match.group(1)
                break
        
        # Extract fine amounts
        fine_patterns = [
            r'fine.*?(\d+.*?(?:million|thousand|lakh).*?rupees?)',
            r'penalty.*?(\d+.*?(?:million|thousand|lakh).*?rupees?)'
        ]
        
        for pattern in fine_patterns:
            match = re.search(pattern, content_lower)
            if match:
                penalty_info['fine'] = match.group(1)
                break
        
        # Check for forfeiture
        if 'forfeiture' in content_lower or 'confiscation' in content_lower:
            penalty_info['forfeiture'] = 'Property forfeiture applicable'
        
        return penalty_info

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        if not text:
            return []
        
        # Clean and tokenize
        text = text.lower()
        words = re.findall(r'\b[a-z]{3,}\b', text)
        
        # Filter stop words and common legal terms
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 
            'one', 'our', 'had', 'but', 'words', 'what', 'were', 'they', 'said', 'each', 
            'which', 'she', 'could', 'make', 'than', 'been', 'call', 'who', 'its', 'now', 
            'find', 'long', 'down', 'day', 'did', 'get', 'has', 'may', 'use', 'shall', 
            'under', 'this', 'such', 'any', 'with', 'person', 'where', 'provided', 
            'means', 'prescribed', 'government'
        }
        
        # Extract meaningful terms
        keywords = []
        for word in words:
            if (word not in stop_words and 
                len(word) > 3 and 
                not word.isdigit()):
                keywords.append(word)
        
        # Return top unique keywords
        return list(set(keywords))[:8]

    def process_all_enhanced_documents(self, folder_path: str) -> Tuple[List[Document], Dict[str, List[Dict]]]:
        """Process all JSON documents in folder with enhanced metadata"""
        all_documents = []
        all_sections = {}
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return [], {}
        
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        if not json_files:
            print(f"⚠️ No JSON files found in {folder_path}")
            return [], {}
        
        print(f"🔄 Processing {len(json_files)} JSON files...")
        
        for filename in json_files:
            file_path = os.path.join(folder_path, filename)
            documents, sections = self.process_json_with_metadata(file_path)
            
            all_documents.extend(documents)
            all_sections[filename] = sections
            
        print(f"✅ Total documents created: {len(all_documents)}")
        return all_documents, all_sections