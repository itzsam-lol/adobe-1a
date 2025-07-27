#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 - Challenge 1a: Optimized PDF Structure Extractor
Complete solution with multilingual support and maximum performance optimization
"""

import fitz  # PyMuPDF
import json
import os
import re
import time
import logging
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import unicodedata
import threading
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class OptimizedMultilingualPDFExtractor:
    """Highly optimized PDF structure extractor with multilingual support"""
    
    def __init__(self):
        # Performance optimization settings
        self.max_workers = min(8, os.cpu_count() or 4)
        self.page_limit = 50  # As per challenge requirement
        self.min_heading_length = 2
        self.max_heading_length = 300
        
        # Font analysis cache
        self.font_cache = {}
        self.pattern_cache = {}
        
        # Multilingual patterns and keywords
        self.multilingual_config = {
            'japanese': {
                'chapter_keywords': ['章', '節', '項', '部', '編', '第', '序', '概要', '要約', '結論'],
                'numbering_patterns': [
                    r'[０-９]{1,3}[．\.]',
                    r'第[０-９一二三四五六七八九十百千]{1,4}[章節項部編]',
                    r'[一二三四五六七八九十百千]{1,4}[．\.]',
                    r'[ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ]+[．\.]?'
                ],
                'title_indicators': ['について', 'に関する', 'の研究', '概論', '入門', '解説', '手引き'],
                'char_range': r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]',
                'threshold': 0.15
            },
            'chinese': {
                'chapter_keywords': ['章', '节', '部分', '第', '序', '概述', '总结', '结论'],
                'numbering_patterns': [
                    r'第[０-９一二三四五六七八九十百千]{1,4}[章节部]',
                    r'[０-９]{1,3}[．\.]',
                    r'[一二三四五六七八九十]{1,4}[．\.]'
                ],
                'title_indicators': ['关于', '研究', '概论', '入门', '指南'],
                'char_range': r'[\u4E00-\u9FFF]',
                'threshold': 0.15
            },
            'korean': {
                'chapter_keywords': ['장', '절', '부', '편', '제', '개요', '요약', '결론'],
                'numbering_patterns': [
                    r'제\s*[０-９]{1,3}\s*장',
                    r'[０-９]{1,3}[．\.]',
                    r'제\s*[０-９]{1,3}\s*절'
                ],
                'title_indicators': ['에 대한', '연구', '개론', '입문', '가이드'],
                'char_range': r'[\uAC00-\uD7AF]',
                'threshold': 0.10
            },
            'english': {
                'chapter_keywords': ['chapter', 'section', 'part', 'introduction', 'conclusion', 'overview', 'summary'],
                'numbering_patterns': [
                    r'\d{1,3}[\.)]',
                    r'[ivxlcdm]+[\.)]',
                    r'[a-z]\)',
                    r'[A-Z]\.'
                ],
                'title_indicators': ['introduction to', 'guide to', 'handbook', 'manual', 'overview of'],
                'char_range': r'[a-zA-Z]',
                'threshold': 0.7
            }
        }
        
        # Pre-compiled regex patterns for performance
        self._compile_patterns()
        
        # Threading lock for thread-safe operations
        self.lock = threading.Lock()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.compiled_patterns = {}
        for lang, config in self.multilingual_config.items():
            self.compiled_patterns[lang] = {
                'numbering': [re.compile(pattern, re.IGNORECASE) for pattern in config['numbering_patterns']],
                'char_range': re.compile(config['char_range']),
                'keywords': re.compile('|'.join(config['chapter_keywords']), re.IGNORECASE)
            }
        
        # Common patterns
        self.noise_patterns = re.compile(
            r'^\s*(page\s+\d+|figure\s+\d+|table\s+\d+|www\.|http|@|\.{3,}|\d+\s*$)\s*$',
            re.IGNORECASE
        )
        self.whitespace_pattern = re.compile(r'\s+')
    
    def detect_language(self, text_sample: str) -> str:
        """Optimized language detection using character frequency analysis"""
        if not text_sample or len(text_sample) < 10:
            return 'english'
        
        # Sample only first 2000 characters for performance
        sample = text_sample[:2000]
        total_chars = len([c for c in sample if not c.isspace()])
        
        if total_chars == 0:
            return 'english'
        
        lang_scores = {}
        for lang, config in self.multilingual_config.items():
            char_pattern = self.compiled_patterns[lang]['char_range']
            matches = len(char_pattern.findall(sample))
            lang_scores[lang] = matches / total_chars
        
        # Return language with highest score above threshold
        max_lang = max(lang_scores.items(), key=lambda x: x[1])
        if max_lang[1] >= self.multilingual_config[max_lang[0]]['threshold']:
            return max_lang[0]
        
        return 'english'
    
    def extract_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Main function to extract structured outline from PDF with optimizations"""
        start_time = time.time()
        
        try:
            # Open PDF with optimized settings
            doc = fitz.open(pdf_path)
            doc_pages = min(len(doc), self.page_limit)
            
            logger.info(f"Processing {Path(pdf_path).name}: {doc_pages} pages")
            
            # Quick language detection from first few pages
            language = self._quick_language_detection(doc)
            logger.info(f"Detected language: {language}")
            
            # Strategy 1: Try built-in TOC first (fastest)
            toc = doc.get_toc()
            if toc and len(toc) > 2:  # Require meaningful TOC
                logger.info("Using built-in table of contents")
                result = self._parse_toc_structure(toc, doc, language)
                doc.close()
                return result
            
            # Strategy 2: Parallel text analysis for performance
            logger.info("Performing advanced text analysis")
            result = self._parallel_text_analysis(doc, language)
            
            doc.close()
            
            processing_time = time.time() - start_time
            logger.info(f"Processed in {processing_time:.2f}s with {len(result['outline'])} headings")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {
                "title": f"Error: {Path(pdf_path).name}",
                "outline": [],
                "error": str(e)
            }
    
    def _quick_language_detection(self, doc) -> str:
        """Quick language detection from first 2 pages"""
        sample_text = ""
        for page_num in range(min(2, len(doc))):
            page = doc[page_num]
            sample_text += page.get_text()[:1000]  # Limit sample size
            if len(sample_text) > 1500:
                break
        
        return self.detect_language(sample_text)
    
    def _parse_toc_structure(self, toc: List, doc, language: str) -> Dict[str, Any]:
        """Parse built-in table of contents with language awareness"""
        title = self._extract_title_from_metadata(doc) or self._extract_title_from_first_page(doc, language)
        outline = []
        
        for entry in toc:
            level, text, page = entry
            
            # Map TOC levels to standard heading levels
            if level == 1:
                heading_level = "H1"
            elif level == 2:
                heading_level = "H2"
            elif level <= 3:
                heading_level = "H3"
            else:
                continue  # Skip deeper levels
            
            # Clean and validate text
            clean_text = self._clean_heading_text(text, language)
            if self._is_valid_heading(clean_text, language):
                outline.append({
                    "level": heading_level,
                    "text": clean_text,
                    "page": max(1, page)  # Ensure page is at least 1
                })
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _parallel_text_analysis(self, doc, language: str) -> Dict[str, Any]:
        """Parallel processing of PDF pages for maximum performance"""
        doc_pages = min(len(doc), self.page_limit)
        
        # Extract all text blocks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create tasks for parallel processing
            future_to_page = {
                executor.submit(self._extract_page_blocks, doc[page], page + 1): page 
                for page in range(doc_pages)
            }
            
            all_blocks = []
            for future in as_completed(future_to_page):
                try:
                    page_blocks = future.result()
                    all_blocks.extend(page_blocks)
                except Exception as e:
                    logger.warning(f"Error processing page {future_to_page[future]}: {e}")
        
        if not all_blocks:
            return {"title": "Empty Document", "outline": []}
        
        # Sort blocks by page and position
        all_blocks.sort(key=lambda x: (x['page'], x.get('y_position', 0)))
        
        # Analyze font characteristics
        font_analysis = self._analyze_fonts_optimized(all_blocks)
        
        # Extract title
        title = self._extract_title_optimized(all_blocks, font_analysis, language)
        
        # Classify headings with advanced heuristics
        outline = self._classify_headings_advanced(all_blocks, font_analysis, title, language)
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _extract_page_blocks(self, page, page_num: int) -> List[Dict]:
        """Extract text blocks from a single page with position information"""
        blocks = []
        try:
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    line_text = ""
                    max_size = 0
                    flags = 0
                    font = ""
                    
                    for span in line["spans"]:
                        span_text = span.get("text", "").strip()
                        if span_text:
                            line_text += span_text + " "
                            max_size = max(max_size, span.get("size", 0))
                            flags |= span.get("flags", 0)
                            if not font and span.get("font"):
                                font = span.get("font", "")
                    
                    line_text = line_text.strip()
                    if (len(line_text) >= self.min_heading_length and 
                        len(line_text) <= self.max_heading_length and
                        not self.noise_patterns.match(line_text)):
                        
                        blocks.append({
                            'text': line_text,
                            'font_size': max_size,
                            'font': font,
                            'flags': flags,
                            'page': page_num,
                            'bbox': line.get("bbox", [0, 0, 0, 0]),
                            'y_position': line.get("bbox", [0, 0, 0, 0])[1] if line.get("bbox") else 0
                        })
        except Exception as e:
            logger.warning(f"Error extracting blocks from page {page_num}: {e}")
        
        return blocks
    
    def _analyze_fonts_optimized(self, text_blocks: List[Dict]) -> Dict[str, Any]:
        """Optimized font analysis with caching"""
        cache_key = f"font_analysis_{len(text_blocks)}"
        
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        font_sizes = [block['font_size'] for block in text_blocks if block['font_size'] > 0]
        
        if not font_sizes:
            return {"body_size": 12, "unique_sizes": [12], "max_size": 12, "size_distribution": {}}
        
        size_counter = Counter(font_sizes)
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        # Determine body text size (most common size, but exclude very large sizes that might be titles)
        filtered_sizes = [size for size, count in size_counter.items() if count > 1]
        body_size = Counter(filtered_sizes).most_common(1)[0][0] if filtered_sizes else min(font_sizes)
        
        analysis = {
            'body_size': body_size,
            'unique_sizes': unique_sizes,
            'size_counter': size_counter,
            'max_size': max(font_sizes),
            'min_size': min(font_sizes),
            'size_distribution': dict(size_counter)
        }
        
        # Cache result
        with self.lock:
            self.font_cache[cache_key] = analysis
        
        return analysis
    
    def _extract_title_optimized(self, text_blocks: List[Dict], font_analysis: Dict, language: str) -> str:
        """Optimized title extraction with multilingual support"""
        title_candidates = []
        max_font_size = font_analysis['max_size']
        body_size = font_analysis['body_size']
        
        # Look for title candidates in first 3 pages
        first_page_blocks = [b for b in text_blocks if b['page'] <= 3][:30]  # Limit for performance
        
        for block in first_page_blocks:
            text = block['text']
            font_size = block['font_size']
            is_bold = bool(block['flags'] & 16)
            
            # Skip if too short or looks like noise
            if len(text) < 5 or len(text) > 150:
                continue
            
            # Calculate title score
            score = 0
            
            # Font size score (larger = better)
            if font_size >= max_font_size * 0.9:
                score += 50
            elif font_size >= body_size * 1.5:
                score += 30
            elif font_size >= body_size * 1.2:
                score += 15
            
            # Position score (earlier = better)
            if block['page'] == 1:
                score += 20
            elif block['page'] == 2:
                score += 10
            
            # Formatting score
            if is_bold:
                score += 15
            
            # Language-specific title indicators
            config = self.multilingual_config.get(language, self.multilingual_config['english'])
            for indicator in config['title_indicators']:
                if indicator.lower() in text.lower():
                    score += 25
                    break
            
            # Text pattern score
            if self._is_title_like_text(text, language):
                score += 20
            
            title_candidates.append((score, text, block['page']))
        
        if title_candidates:
            title_candidates.sort(reverse=True, key=lambda x: (x[0], -x[2]))  # Sort by score, then by earlier page
            best_title = title_candidates[0][1]
            return self._clean_heading_text(best_title, language)
        
        # Fallback: try metadata
        return "Untitled Document"
    
    def _classify_headings_advanced(self, text_blocks: List[Dict], font_analysis: Dict, title: str, language: str) -> List[Dict]:
        """Advanced heading classification with multilingual support"""
        outline = []
        body_size = font_analysis['body_size']
        max_size = font_analysis['max_size']
        
        # Dynamic thresholds based on document characteristics
        size_range = max_size - body_size
        
        if size_range > 8:  # Large range - use absolute thresholds
            h1_threshold = body_size + size_range * 0.6
            h2_threshold = body_size + size_range * 0.4
            h3_threshold = body_size + size_range * 0.2
        else:  # Small range - use relative thresholds
            h1_threshold = body_size * 1.4
            h2_threshold = body_size * 1.2
            h3_threshold = body_size * 1.05
        
        config = self.multilingual_config.get(language, self.multilingual_config['english'])
        
        for block in text_blocks:
            text = block['text']
            font_size = block['font_size']
            is_bold = bool(block['flags'] & 16)
            
            # Skip title and invalid text
            if (text == title or 
                len(text) < self.min_heading_length or 
                len(text) > self.max_heading_length or
                self._is_likely_body_text(text, language)):
                continue
            
            # Determine heading level using multiple criteria
            heading_level = None
            confidence = 0
            
            # Font size criteria
            if font_size >= h1_threshold:
                heading_level = "H1"
                confidence += 40
            elif font_size >= h2_threshold:
                heading_level = "H2"
                confidence += 30
            elif font_size >= h3_threshold:
                heading_level = "H3"
                confidence += 20
            
            # Bold formatting bonus
            if is_bold and font_size >= body_size:
                if not heading_level:
                    heading_level = "H3"
                confidence += 20
            
            # Pattern matching bonus
            pattern_score = self._calculate_pattern_score(text, language)
            confidence += pattern_score
            
            # Structure bonus (numbering, keywords)
            if self._has_structural_indicators(text, language):
                if not heading_level and font_size >= body_size * 0.95:
                    heading_level = "H3"
                confidence += 25
            
            # Only include if confidence is high enough
            if heading_level and confidence >= 25:
                clean_text = self._clean_heading_text(text, language)
                if self._is_valid_heading(clean_text, language):
                    outline.append({
                        "level": heading_level,
                        "text": clean_text,
                        "page": block['page']
                    })
        
        # Post-processing: remove duplicates and sort
        outline = self._post_process_outline(outline)
        
        return outline
    
    def _calculate_pattern_score(self, text: str, language: str) -> int:
        """Calculate pattern matching score for heading detection"""
        score = 0
        config = self.multilingual_config.get(language, self.multilingual_config['english'])
        
        # Check numbering patterns
        for pattern in self.compiled_patterns[language]['numbering']:
            if pattern.match(text.strip()):
                score += 30
                break
        
        # Check keyword patterns
        if self.compiled_patterns[language]['keywords'].search(text):
            score += 20
        
        # Check title case (for languages that use it)
        if language == 'english':
            words = text.split()
            if len(words) >= 2:
                capitalized = sum(1 for word in words if word and word[0].isupper())
                if capitalized / len(words) >= 0.6:
                    score += 15
        
        return score
    
    def _has_structural_indicators(self, text: str, language: str) -> bool:
        """Check for structural indicators of headings"""
        config = self.multilingual_config.get(language, self.multilingual_config['english'])
        
        # Check for chapter/section keywords
        for keyword in config['chapter_keywords']:
            if keyword.lower() in text.lower():
                return True
        
        # Check for numbering at start
        for pattern in self.compiled_patterns[language]['numbering']:
            if pattern.match(text.strip()):
                return True
        
        return False
    
    def _is_likely_body_text(self, text: str, language: str) -> bool:
        """Enhanced body text detection with language awareness"""
        # Length check
        if len(text) > 200:
            return True
        
        # Sentence structure check (multiple sentences)
        if text.count('.') > 1 or text.count('。') > 1:  # Japanese period
            return True
        
        # Word ratio analysis for English
        if language == 'english':
            words = text.split()
            if len(words) > 8:
                lowercase_ratio = sum(1 for word in words if word and word[0].islower()) / len(words)
                if lowercase_ratio > 0.6:
                    return True
        
        # Common body text patterns
        body_indicators = [
            'however', 'therefore', 'furthermore', 'in addition', 'for example',
            'こんにちは', 'である', 'について', 'における',  # Japanese
            '因此', '然而', '例如', '此外',  # Chinese
            '그러나', '따라서', '예를 들어'  # Korean
        ]
        
        text_lower = text.lower()
        for indicator in body_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    def _is_title_like_text(self, text: str, language: str) -> bool:
        """Check if text looks like a document title"""
        config = self.multilingual_config.get(language, self.multilingual_config['english'])
        
        text_lower = text.lower()
        for indicator in config['title_indicators']:
            if indicator in text_lower:
                return True
        
        return False
    
    def _is_valid_heading(self, text: str, language: str) -> bool:
        """Validate heading text with language awareness"""
        text = text.strip()
        
        # Length check
        if len(text) < self.min_heading_length or len(text) > self.max_heading_length:
            return False
        
        # Must contain some alphabetic characters
        if not any(c.isalpha() or ord(c) > 127 for c in text):  # Include non-ASCII chars
            return False
        
        # Skip obvious noise patterns
        if self.noise_patterns.match(text):
            return False
        
        return True
    
    def _clean_heading_text(self, text: str, language: str) -> str:
        """Clean heading text with language-specific rules"""
        text = text.strip()
        
        # Language-specific cleaning
        if language == 'japanese':
            # Remove Japanese numbering
            text = re.sub(r'^[０-９]{1,3}[．\.]?\s*', '', text)
            text = re.sub(r'^第[０-９一二三四五六七八九十百千]{1,4}[章節項部編]\s*', '', text)
            text = re.sub(r'^[一二三四五六七八九十百千]{1,4}[．\.]?\s*', '', text)
        elif language == 'chinese':
            # Remove Chinese numbering
            text = re.sub(r'^第[０-９一二三四五六七八九十百千]{1,4}[章节部]\s*', '', text)
            text = re.sub(r'^[０-９]{1,3}[．\.]?\s*', '', text)
        elif language == 'korean':
            # Remove Korean numbering
            text = re.sub(r'^제\s*[０-９]{1,3}\s*[장절]\s*', '', text)
        else:  # English and others
            # Remove common English numbering
            text = re.sub(r'^\d{1,3}[\.)\s]+', '', text)
            text = re.sub(r'^[ivxlcdm]+[\.)\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'^(chapter|section|part)\s+\d+\.?\s*', '', text, flags=re.IGNORECASE)
        
        # Common cleaning
        text = self.whitespace_pattern.sub(' ', text)
        text = text.rstrip('.-:;')
        
        return text.strip()
    
    def _extract_title_from_metadata(self, doc) -> Optional[str]:
        """Extract title from PDF metadata"""
        try:
            metadata = doc.metadata
            title = metadata.get('title', '').strip()
            if title and len(title) > 3 and len(title) < 200:
                return title
        except:
            pass
        return None
    
    def _extract_title_from_first_page(self, doc, language: str) -> str:
        """Extract title from first page content"""
        try:
            if len(doc) > 0:
                first_page = doc[0]
                text_dict = first_page.get_text("dict")
                
                max_size = 0
                title_text = ""
                
                for block in text_dict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            size = span.get("size", 0)
                            
                            if (text and len(text) > 5 and len(text) < 150 and 
                                size > max_size and size > 14):
                                max_size = size
                                title_text = text
                
                if title_text:
                    return self._clean_heading_text(title_text, language)
        except:
            pass
        
        return "Untitled Document"
    
    def _post_process_outline(self, outline: List[Dict]) -> List[Dict]:
        """Post-process outline to remove duplicates and improve quality"""
        if not outline:
            return outline
        
        # Remove duplicates based on text similarity
        seen_texts = set()
        deduplicated = []
        
        for item in outline:
            text_key = item['text'].lower().strip()
            if text_key not in seen_texts and len(text_key) > 0:
                seen_texts.add(text_key)
                deduplicated.append(item)
        
        # Sort by page number, then by level
        level_order = {"H1": 1, "H2": 2, "H3": 3}
        deduplicated.sort(key=lambda x: (x['page'], level_order.get(x['level'], 4)))
        
        # Limit to reasonable number of headings for performance
        return deduplicated[:100]  # Max 100 headings


def process_pdfs_optimized():
    """Main processing function with error handling and performance monitoring"""
    start_time = time.time()
    
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Initialize extractor
    extractor = OptimizedMultilingualPDFExtractor()
    
    # Process files with performance monitoring
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        file_start = time.time()
        
        try:
            logger.info(f"Processing {pdf_file.name}...")
            
            # Extract structure
            result = extractor.extract_pdf_structure(str(pdf_file))
            
            # Remove internal fields before output
            output_result = {
                "title": result["title"],
                "outline": result["outline"]
            }
            
            # Save JSON output
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_result, f, indent=2, ensure_ascii=False)
            
            file_time = time.time() - file_start
            logger.info(f"✅ {output_file.name}: {len(result['outline'])} headings in {file_time:.2f}s")
            successful += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_file.name}: {e}")
            # Create minimal output for failed files
            error_output = {
                "title": f"Processing Error: {pdf_file.name}",
                "outline": []
            }
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_output, f, indent=2)
            failed += 1
    
    total_time = time.time() - start_time
    logger.info(f"🎉 Processing completed: {successful} successful, {failed} failed in {total_time:.2f}s")


if __name__ == "__main__":
    logger.info("🚀 Starting Optimized PDF Structure Extractor - Challenge 1a")
    process_pdfs_optimized()
    logger.info("✅ Processing completed")
