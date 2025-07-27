#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 - Challenge 1a
PDF Structure Extractor with Multilingual Support
Author: Your Team Name
"""

import fitz  # PyMuPDF
import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import logging
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultilingualPDFStructureExtractor:
    def __init__(self):
        self.min_heading_length = 2
        self.max_heading_length = 300
        self.common_fonts = {}
        
        # Multilingual configuration
        self.language_config = {
            'japanese': {
                'chapter_keywords': ['章', '節', '項', '部', '編', '第', '序', '概要', '要約', '結論', 'はじめに'],
                'numbering_patterns': [
                    r'[０-９]{1,3}[．\.]',
                    r'第[０-９一二三四五六七八九十百千]{1,4}[章節項部編]',
                    r'[一二三四五六七八九十百千]{1,4}[．\.]',
                    r'[ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ]+[．\.]?'
                ],
                'char_range': r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]',
                'threshold': 0.1
            },
            'chinese': {
                'chapter_keywords': ['章', '节', '部分', '第', '序', '概述', '总结', '结论', '前言', '导论'],
                'numbering_patterns': [
                    r'第[０-９一二三四五六七八九十百千]{1,4}[章节部]',
                    r'[０-９]{1,3}[．\.]',
                    r'[一二三四五六七八九十]{1,4}[．\.]'
                ],
                'char_range': r'[\u4E00-\u9FFF]',
                'threshold': 0.1
            },
            'korean': {
                'chapter_keywords': ['장', '절', '부', '편', '제', '개요', '요약', '결론', '서론', '도입'],
                'numbering_patterns': [
                    r'제\s*[０-９]{1,3}\s*장',
                    r'[０-９]{1,3}[．\.]',
                    r'제\s*[０-９]{1,3}\s*절'
                ],
                'char_range': r'[\uAC00-\uD7AF]',
                'threshold': 0.08
            },
            'arabic': {
                'chapter_keywords': ['الفصل', 'الباب', 'الجزء', 'المقدمة', 'الخلاصة', 'النتائج', 'المحتويات'],
                'numbering_patterns': [
                    r'الفصل\s+[٠-٩0-9]+',
                    r'[٠-٩0-9]+[\.)]',
                    r'الباب\s+[٠-٩0-9]+'
                ],
                'char_range': r'[\u0600-\u06FF\u0750-\u077F]',
                'threshold': 0.1
            },
            'hindi': {
                'chapter_keywords': ['अध्याय', 'भाग', 'खंड', 'प्रस्तावना', 'निष्कर्ष', 'सारांश', 'विषय'],
                'numbering_patterns': [
                    r'अध्याय\s+[०-९0-9]+',
                    r'[०-९0-9]+[\.)]',
                    r'भाग\s+[०-९0-9]+'
                ],
                'char_range': r'[\u0900-\u097F]',
                'threshold': 0.08
            },
            'english': {
                'chapter_keywords': ['chapter', 'section', 'part', 'introduction', 'conclusion', 'overview', 'summary'],
                'numbering_patterns': [
                    r'\d{1,3}[\.)]',
                    r'[ivxlcdm]+[\.)]',
                    r'[a-z]\)',
                    r'[A-Z]\.'
                ],
                'char_range': r'[a-zA-Z]',
                'threshold': 0.6
            },
            # European languages
            'french': {
                'chapter_keywords': ['chapitre', 'section', 'partie', 'introduction', 'conclusion', 'résumé', 'sommaire'],
                'numbering_patterns': [
                    r'\d{1,3}[\.)]',
                    r'[ivxlcdm]+[\.)]'
                ],
                'char_range': r'[a-zA-ZàâäéèêëïîôöùûüÿñçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÑÇ]',
                'threshold': 0.6
            },
            'german': {
                'chapter_keywords': ['kapitel', 'abschnitt', 'teil', 'einleitung', 'schluss', 'zusammenfassung', 'inhalt'],
                'numbering_patterns': [
                    r'\d{1,3}[\.)]',
                    r'[ivxlcdm]+[\.)]'
                ],
                'char_range': r'[a-zA-ZäöüßÄÖÜ]',
                'threshold': 0.6
            },
            'spanish': {
                'chapter_keywords': ['capítulo', 'sección', 'parte', 'introducción', 'conclusión', 'resumen', 'índice'],
                'numbering_patterns': [
                    r'\d{1,3}[\.)]',
                    r'[ivxlcdm]+[\.)]'
                ],
                'char_range': r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]',
                'threshold': 0.6
            },
            'italian': {
                'chapter_keywords': ['capitolo', 'sezione', 'parte', 'introduzione', 'conclusione', 'riassunto', 'indice'],
                'numbering_patterns': [
                    r'\d{1,3}[\.)]',
                    r'[ivxlcdm]+[\.)]'
                ],
                'char_range': r'[a-zA-ZàèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ]',
                'threshold': 0.6
            },
            'portuguese': {
                'chapter_keywords': ['capítulo', 'seção', 'parte', 'introdução', 'conclusão', 'resumo', 'índice'],
                'numbering_patterns': [
                    r'\d{1,3}[\.)]',
                    r'[ivxlcdm]+[\.)]'
                ],
                'char_range': r'[a-zA-ZãáàâéêíóôõúçÃÁÀÂÉÊÍÓÔÕÚÇ]',
                'threshold': 0.6
            },
            'russian': {
                'chapter_keywords': ['глава', 'раздел', 'часть', 'введение', 'заключение', 'резюме', 'содержание'],
                'numbering_patterns': [
                    r'\d{1,3}[\.)]',
                    r'глава\s+\d+'
                ],
                'char_range': r'[\u0400-\u04FF]',
                'threshold': 0.1
            }
        }
        
        # Compile patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.compiled_patterns = {}
        for lang, config in self.language_config.items():
            self.compiled_patterns[lang] = {
                'numbering': [re.compile(pattern, re.IGNORECASE | re.UNICODE) for pattern in config['numbering_patterns']],
                'char_range': re.compile(config['char_range']),
                'keywords': re.compile('|'.join(re.escape(kw) for kw in config['chapter_keywords']), re.IGNORECASE | re.UNICODE)
            }
    
    def detect_language(self, text_sample):
        """Detect the primary language of the document"""
        if not text_sample or len(text_sample) < 10:
            return 'english'
        
        # Sample first 2000 characters for performance
        sample = text_sample[:2000]
        total_chars = len([c for c in sample if not c.isspace() and c.isalnum()])
        
        if total_chars == 0:
            return 'english'
        
        lang_scores = {}
        for lang, config in self.language_config.items():
            char_pattern = self.compiled_patterns[lang]['char_range']
            matches = len(char_pattern.findall(sample))
            score = matches / total_chars if total_chars > 0 else 0
            lang_scores[lang] = score
        
        # Find language with highest score above threshold
        max_lang = max(lang_scores.items(), key=lambda x: x[1])
        if max_lang[1] >= self.language_config[max_lang[0]]['threshold']:
            logger.info(f"Detected language: {max_lang[0]} (confidence: {max_lang[1]:.2f})")
            return max_lang[0]
        
        logger.info("Language detection uncertain, defaulting to English")
        return 'english'
    
    def extract_pdf_structure(self, pdf_path):
        """Main function to extract structured outline from PDF"""
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Processing PDF: {pdf_path} ({len(doc)} pages)")
            
            # Quick language detection
            language = self._detect_language_from_doc(doc)
            
            # Strategy 1: Try to use built-in table of contents
            toc = doc.get_toc()
            if toc and len(toc) > 0:
                logger.info("Using built-in table of contents")
                return self.parse_toc_structure(toc, doc, language)
            
            # Strategy 2: Analyze text formatting for heading detection
            logger.info("Analyzing text formatting for heading detection")
            return self.analyze_text_formatting(doc, language)
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return {
                "title": f"Error processing {Path(pdf_path).name}",
                "outline": []
            }
        finally:
            if 'doc' in locals():
                doc.close()
    
    def _detect_language_from_doc(self, doc):
        """Quick language detection from first few pages"""
        sample_text = ""
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()[:1000]  # Limit for performance
            sample_text += page_text
            if len(sample_text) > 2000:
                break
        
        return self.detect_language(sample_text)
    
    def parse_toc_structure(self, toc, doc, language):
        """Parse built-in table of contents with language awareness"""
        title = self.extract_title_from_doc(doc, language)
        outline = []
        
        for entry in toc:
            level, text, page = entry
            # Map TOC levels to heading levels
            if level == 1:
                heading_level = "H1"
            elif level == 2:
                heading_level = "H2"
            elif level == 3:
                heading_level = "H3"
            else:
                continue  # Skip deeper levels
            
            # Clean up text with language awareness
            clean_text = self.clean_heading_text(text, language)
            if self.is_valid_heading(clean_text, language):
                outline.append({
                    "level": heading_level,
                    "text": clean_text,
                    "page": max(1, page)
                })
        
        return {
            "title": title,
            "outline": outline
        }
    
    def analyze_text_formatting(self, doc, language):
        """Analyze font characteristics to detect headings with language support"""
        # Extract all text blocks with formatting information
        text_blocks = self.extract_text_blocks(doc)
        
        if not text_blocks:
            return {
                "title": "Untitled Document",
                "outline": []
            }
        
        # Analyze font characteristics
        font_analysis = self.analyze_fonts(text_blocks)
        
        # Extract title with language awareness
        title = self.extract_title(text_blocks, font_analysis, language)
        
        # Classify headings with multilingual support
        outline = self.classify_headings(text_blocks, font_analysis, title, language)
        
        return {
            "title": title,
            "outline": outline
        }
    
    def extract_text_blocks(self, doc):
        """Extract text blocks with formatting information"""
        text_blocks = []
        
        for page_num in range(min(len(doc), 50)):  # Limit to 50 pages as per requirement
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_fonts = []
                        line_sizes = []
                        line_flags = []
                        
                        for span in line["spans"]:
                            span_text = span.get("text", "").strip()
                            if span_text:
                                line_text += span_text + " "
                                line_fonts.append(span.get("font", ""))
                                line_sizes.append(span.get("size", 0))
                                line_flags.append(span.get("flags", 0))
                        
                        line_text = line_text.strip()
                        if line_text and len(line_text) >= self.min_heading_length:
                            text_blocks.append({
                                'text': line_text,
                                'font_size': max(line_sizes) if line_sizes else 0,
                                'font': line_fonts[0] if line_fonts else "",
                                'flags': max(line_flags) if line_flags else 0,
                                'page': page_num + 1,
                                'bbox': line.get("bbox", [0, 0, 0, 0])
                            })
        
        return text_blocks
    
    def analyze_fonts(self, text_blocks):
        """Analyze font characteristics across the document"""
        font_sizes = [block['font_size'] for block in text_blocks if block['font_size'] > 0]
        fonts = [block['font'] for block in text_blocks if block['font']]
        
        # Get font size statistics
        size_counter = Counter(font_sizes)
        font_counter = Counter(fonts)
        
        # Determine common body text size (most frequent)
        body_size = size_counter.most_common(1)[0][0] if size_counter else 12
        
        # Get unique font sizes in descending order
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        return {
            'body_size': body_size,
            'unique_sizes': unique_sizes,
            'size_counter': size_counter,
            'font_counter': font_counter,
            'max_size': max(font_sizes) if font_sizes else 0,
            'min_size': min(font_sizes) if font_sizes else 0
        }
    
    def extract_title(self, text_blocks, font_analysis, language):
        """Extract document title with multilingual support"""
        # Look for title on first few pages with largest font size
        title_candidates = []
        
        for block in text_blocks[:25]:  # Check first 25 text blocks
            if (block['page'] <= 3 and 
                block['font_size'] >= font_analysis['body_size'] * 1.2 and
                len(block['text']) >= 3 and
                len(block['text']) <= 150):
                
                # Score based on font size, position, and formatting
                score = (
                    block['font_size'] * 2 +
                    (1 if block['flags'] & 16 else 0) * 10 +  # Bold
                    (1 if block['page'] == 1 else 0) * 20 +   # First page
                    (1 if self.is_title_like(block['text'], language) else 0) * 15
                )
                
                title_candidates.append((score, block['text']))
        
        if title_candidates:
            title_candidates.sort(reverse=True)
            return self.clean_heading_text(title_candidates[0][1], language)
        
        return "Untitled Document"
    
    def classify_headings(self, text_blocks, font_analysis, title, language):
        """Classify text blocks into heading levels with multilingual support"""
        outline = []
        body_size = font_analysis['body_size']
        
        # Define size thresholds for heading levels
        h1_threshold = body_size * 1.4
        h2_threshold = body_size * 1.2
        h3_threshold = body_size * 1.05
        
        for block in text_blocks:
            text = block['text']
            font_size = block['font_size']
            is_bold = bool(block['flags'] & 16)
            
            # Skip if it's the title or too short/long
            if (text == title or 
                len(text) < self.min_heading_length or 
                len(text) > self.max_heading_length):
                continue
            
            # Skip common non-heading patterns
            if self.is_likely_body_text(text, language):
                continue
            
            # Determine heading level with multilingual patterns
            heading_level = None
            confidence = 0
            
            # Font size based classification
            if font_size >= h1_threshold or (font_size >= body_size * 1.1 and is_bold):
                if self.is_heading_like(text, "H1", language):
                    heading_level = "H1"
                    confidence += 30
            elif font_size >= h2_threshold or (font_size >= body_size and is_bold):
                if self.is_heading_like(text, "H2", language):
                    heading_level = "H2"
                    confidence += 25
            elif font_size >= h3_threshold:
                if self.is_heading_like(text, "H3", language):
                    heading_level = "H3"
                    confidence += 20
            
            # Pattern-based enhancement
            pattern_confidence = self._check_multilingual_patterns(text, language)
            confidence += pattern_confidence
            
            # If pattern suggests heading but font didn't, reconsider
            if not heading_level and pattern_confidence > 20 and font_size >= body_size * 0.9:
                heading_level = "H3"
                confidence += pattern_confidence
            
            if heading_level and confidence > 15 and self.is_valid_heading(text, language):
                clean_text = self.clean_heading_text(text, language)
                outline.append({
                    "level": heading_level,
                    "text": clean_text,
                    "page": block['page']
                })
        
        # Remove duplicates and sort by page
        outline = self.deduplicate_outline(outline)
        outline.sort(key=lambda x: (x['page'], x['level']))
        
        return outline
    
    def _check_multilingual_patterns(self, text, language):
        """Check for language-specific heading patterns"""
        confidence = 0
        config = self.language_config.get(language, self.language_config['english'])
        
        # Check numbering patterns
        for pattern in self.compiled_patterns[language]['numbering']:
            if pattern.match(text.strip()):
                confidence += 25
                break
        
        # Check keyword patterns
        if self.compiled_patterns[language]['keywords'].search(text):
            confidence += 20
        
        return confidence
    
    def is_heading_like(self, text, level, language):
        """Check if text looks like a heading with multilingual support"""
        # Check multilingual patterns first
        if self._check_multilingual_patterns(text, language) > 15:
            return True
        
        # Original English patterns
        heading_patterns = [
            r'^\d+\.?\s+',  # "1. Introduction" or "1 Introduction"
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # "Title Case Words"
            r'^[A-Z\s]+$',  # "ALL CAPS"
            r'^(Chapter|Section|Part)\s+\d+',  # "Chapter 1", "Section 2"
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text.strip(), re.UNICODE):
                return True
        
        # Check for title case (for languages that use it)
        if language in ['english', 'french', 'german', 'spanish', 'italian', 'portuguese']:
            words = text.split()
            if len(words) >= 2:
                capitalized = sum(1 for word in words if word and len(word) > 0 and word[0].isupper())
                if capitalized / len(words) >= 0.6:
                    return True
        
        return False
    
    def is_likely_body_text(self, text, language):
        """Enhanced body text detection with language awareness"""
        # Skip very long paragraphs
        if len(text) > 250:
            return True
        
        # Skip text with multiple sentences
        sentence_endings = ['.', '。', '！', '？', '!', '?']
        sentence_count = sum(text.count(ending) for ending in sentence_endings)
        if sentence_count > 1:
            return True
        
        # Language-specific body text patterns
        body_indicators = {
            'english': ['however', 'therefore', 'furthermore', 'in addition', 'for example', 'the', 'and', 'or', 'but'],
            'japanese': ['である', 'について', 'における', 'として', 'により', 'また', 'さらに'],
            'chinese': ['因此', '然而', '例如', '此外', '而且', '另外', '同时'],
            'korean': ['그러나', '따라서', '예를 들어', '또한', '게다가'],
            'arabic': ['ولكن', 'لذلك', 'على سبيل المثال', 'بالإضافة إلى ذلك'],
            'hindi': ['किंतु', 'इसलिए', 'उदाहरण के लिए', 'इसके अतिरिक्त'],
            'french': ['cependant', 'par conséquent', 'par exemple', 'de plus'],
            'german': ['jedoch', 'daher', 'zum Beispiel', 'außerdem'],
            'spanish': ['sin embargo', 'por lo tanto', 'por ejemplo', 'además'],
            'italian': ['tuttavia', 'pertanto', 'per esempio', 'inoltre'],
            'portuguese': ['no entanto', 'portanto', 'por exemplo', 'além disso'],
            'russian': ['однако', 'поэтому', 'например', 'кроме того']
        }
        
        indicators = body_indicators.get(language, body_indicators['english'])
        text_lower = text.lower()
        
        indicator_count = sum(1 for indicator in indicators if indicator in text_lower)
        if indicator_count > 2:
            return True
        
        # For Latin-based languages, check lowercase ratio
        if language in ['english', 'french', 'german', 'spanish', 'italian', 'portuguese']:
            words = text.split()
            if len(words) > 8:
                lowercase_ratio = sum(1 for word in words if word and word[0].islower()) / len(words)
                if lowercase_ratio > 0.7:
                    return True
        
        return False
    
    def is_title_like(self, text, language):
        """Check if text looks like a document title with language awareness"""
        title_indicators = {
            'english': ['introduction', 'overview', 'guide', 'manual', 'handbook', 'report', 'analysis', 'study'],
            'japanese': ['概要', '序論', '入門', 'ガイド', '手引き', '報告', '研究', '分析'],
            'chinese': ['概述', '介绍', '指南', '手册', '报告', '分析', '研究'],
            'korean': ['개요', '소개', '가이드', '안내서', '보고서', '분석', '연구'],
            'arabic': ['مقدمة', 'نظرة عامة', 'دليل', 'تقرير', 'تحليل', 'دراسة'],
            'hindi': ['परिचय', 'अवलोकन', 'गाइड', 'रिपोर्ट', 'विश्लेषण', 'अध्ययन'],
            'french': ['introduction', 'aperçu', 'guide', 'rapport', 'analyse', 'étude'],
            'german': ['einführung', 'überblick', 'leitfaden', 'bericht', 'analyse', 'studie'],
            'spanish': ['introducción', 'resumen', 'guía', 'informe', 'análisis', 'estudio'],
            'italian': ['introduzione', 'panoramica', 'guida', 'rapporto', 'analisi', 'studio'],
            'portuguese': ['introdução', 'visão geral', 'guia', 'relatório', 'análise', 'estudo'],
            'russian': ['введение', 'обзор', 'руководство', 'отчет', 'анализ', 'исследование']
        }
        
        indicators = title_indicators.get(language, title_indicators['english'])
        text_lower = text.lower()
        
        return any(indicator in text_lower for indicator in indicators)
    
    def is_valid_heading(self, text, language):
        """Validate if text is a proper heading with language awareness"""
        text = text.strip()
        
        # Check length constraints
        if len(text) < self.min_heading_length or len(text) > self.max_heading_length:
            return False
        
        # Must contain some meaningful characters
        if language in ['japanese', 'chinese', 'korean', 'arabic', 'hindi']:
            # For non-Latin scripts, check for script-specific characters
            config = self.language_config.get(language, {})
            char_pattern = self.compiled_patterns.get(language, {}).get('char_range')
            if char_pattern and not char_pattern.search(text):
                # If no script-specific chars, check for numbers/Latin
                if not any(c.isalnum() for c in text):
                    return False
        else:
            # For Latin-based languages
            if not any(c.isalpha() for c in text):
                return False
        
        # Skip common noise patterns (enhanced for multilingual)
        noise_patterns = [
            r'^\d+$',  # Pure numbers
            r'^page\s+\d+',  # Page numbers
            r'^figure\s+\d+',  # Figure captions
            r'^table\s+\d+',  # Table captions
            r'^www\.',  # URLs
            r'@',  # Email addresses
            r'^\s*[\.]{3,}\s*$',  # Dots only
            r'^\s*[-_=]{3,}\s*$'  # Lines/underscores
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, text, re.IGNORECASE | re.UNICODE):
                return False
        
        return True
    
    def clean_heading_text(self, text, language):
        """Clean and normalize heading text with language-specific rules"""
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
        elif language == 'arabic':
            # Remove Arabic numbering
            text = re.sub(r'^الفصل\s+[٠-٩0-9]+\s*', '', text)
            text = re.sub(r'^[٠-٩0-9]+[\.)\s]+', '', text)
        elif language == 'hindi':
            # Remove Hindi numbering
            text = re.sub(r'^अध्याय\s+[०-९0-9]+\s*', '', text)
            text = re.sub(r'^[०-९0-9]+[\.)\s]+', '', text)
        else:
            # Default cleaning for Latin-based languages
            text = re.sub(r'^\d+[\.)\s]+', '', text)
            text = re.sub(r'^[ivxlcdm]+[\.)\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'^(chapter|section|part|chapitre|kapitel|capítulo|capitolo)\s+\d+\.?\s*', '', text, flags=re.IGNORECASE)
        
        # Common cleaning for all languages
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.rstrip('.-:;')  # Remove trailing punctuation
        
        return text.strip()
    
    def deduplicate_outline(self, outline):
        """Remove duplicate headings"""
        seen = set()
        deduplicated = []
        
        for item in outline:
            # Create a normalized key for comparison
            normalized_text = unicodedata.normalize('NFKC', item['text'].lower().strip())
            key = (normalized_text, item['level'])
            if key not in seen:
                seen.add(key)
                deduplicated.append(item)
        
        return deduplicated
    
    def extract_title_from_doc(self, doc, language):
        """Extract title from document metadata or first page with language support"""
        # Try metadata first
        try:
            metadata = doc.metadata
            if metadata.get('title'):
                title = metadata['title'].strip()
                if len(title) > 3 and len(title) < 200:
                    return self.clean_heading_text(title, language)
        except:
            pass
        
        # Fall back to text analysis with language awareness
        if len(doc) > 0:
            first_page = doc[0]
            text_dict = first_page.get_text("dict")
            
            # Look for large text on first page
            max_size = 0
            title_candidate = ""
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            size = span.get("size", 0)
                            if (text and len(text) > 3 and len(text) < 150 and 
                                size > max_size and size > 12):
                                # Additional validation for non-noise text
                                if self.is_valid_heading(text, language):
                                    max_size = size
                                    title_candidate = text
            
            if title_candidate:
                return self.clean_heading_text(title_candidate, language)
        
        return "Untitled Document"

def process_pdfs():
    """Main processing function"""
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    # Initialize multilingual extractor
    extractor = MultilingualPDFStructureExtractor()
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing {pdf_file.name}...")
            
            # Extract structure with multilingual support
            result = extractor.extract_pdf_structure(str(pdf_file))
            
            # Save JSON output
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated {output_file.name} with {len(result['outline'])} headings")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            # Create minimal output for failed files
            error_output = {
                "title": f"Error processing {pdf_file.name}",
                "outline": []
            }
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_output, f, indent=2)

if __name__ == "__main__":
    logger.info("Starting Multilingual PDF Structure Extractor - Challenge 1a")
    process_pdfs()
    logger.info("Processing completed")
