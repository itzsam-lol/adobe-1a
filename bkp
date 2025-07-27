#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 - Challenge 1a
PDF Structure Extractor
Author: Your Team Name
"""

import fitz  # PyMuPDF
import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFStructureExtractor:
    def __init__(self):
        self.min_heading_length = 3
        self.max_heading_length = 200
        self.common_fonts = {}
        
    def extract_pdf_structure(self, pdf_path):
        """Main function to extract structured outline from PDF"""
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Processing PDF: {pdf_path} ({len(doc)} pages)")
            
            # Strategy 1: Try to use built-in table of contents
            toc = doc.get_toc()
            if toc and len(toc) > 0:
                logger.info("Using built-in table of contents")
                return self.parse_toc_structure(toc, doc)
            
            # Strategy 2: Analyze text formatting for heading detection
            logger.info("Analyzing text formatting for heading detection")
            return self.analyze_text_formatting(doc)
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return {
                "title": f"Error processing {Path(pdf_path).name}",
                "outline": []
            }
        finally:
            if 'doc' in locals():
                doc.close()
    
    def parse_toc_structure(self, toc, doc):
        """Parse built-in table of contents"""
        title = self.extract_title_from_doc(doc)
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
            
            # Clean up text
            clean_text = self.clean_heading_text(text)
            if self.is_valid_heading(clean_text):
                outline.append({
                    "level": heading_level,
                    "text": clean_text,
                    "page": page
                })
        
        return {
            "title": title,
            "outline": outline
        }
    
    def analyze_text_formatting(self, doc):
        """Analyze font characteristics to detect headings"""
        # Extract all text blocks with formatting information
        text_blocks = self.extract_text_blocks(doc)
        
        if not text_blocks:
            return {
                "title": "Untitled Document",
                "outline": []
            }
        
        # Analyze font characteristics
        font_analysis = self.analyze_fonts(text_blocks)
        
        # Extract title
        title = self.extract_title(text_blocks, font_analysis)
        
        # Classify headings
        outline = self.classify_headings(text_blocks, font_analysis, title)
        
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
    
    def extract_title(self, text_blocks, font_analysis):
        """Extract document title"""
        # Look for title on first few pages with largest font size
        title_candidates = []
        
        for block in text_blocks[:20]:  # Check first 20 text blocks
            if (block['page'] <= 3 and 
                block['font_size'] >= font_analysis['body_size'] * 1.2 and
                len(block['text']) >= 5 and
                len(block['text']) <= 100):
                
                # Score based on font size, position, and formatting
                score = (
                    block['font_size'] * 2 +
                    (1 if block['flags'] & 16 else 0) * 10 +  # Bold
                    (1 if block['page'] == 1 else 0) * 20 +   # First page
                    (1 if self.is_title_like(block['text']) else 0) * 15
                )
                
                title_candidates.append((score, block['text']))
        
        if title_candidates:
            title_candidates.sort(reverse=True)
            return self.clean_heading_text(title_candidates[0][1])
        
        return "Untitled Document"
    
    def classify_headings(self, text_blocks, font_analysis, title):
        """Classify text blocks into heading levels"""
        outline = []
        body_size = font_analysis['body_size']
        
        # Define size thresholds for heading levels
        h1_threshold = body_size * 1.4
        h2_threshold = body_size * 1.2
        h3_threshold = body_size * 1.1
        
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
            if self.is_likely_body_text(text):
                continue
            
            # Determine heading level
            heading_level = None
            
            if font_size >= h1_threshold or (font_size >= body_size * 1.1 and is_bold):
                if self.is_heading_like(text, "H1"):
                    heading_level = "H1"
            elif font_size >= h2_threshold or (font_size >= body_size and is_bold):
                if self.is_heading_like(text, "H2"):
                    heading_level = "H2"
            elif font_size >= h3_threshold:
                if self.is_heading_like(text, "H3"):
                    heading_level = "H3"
            
            if heading_level and self.is_valid_heading(text):
                clean_text = self.clean_heading_text(text)
                outline.append({
                    "level": heading_level,
                    "text": clean_text,
                    "page": block['page']
                })
        
        # Remove duplicates and sort by page
        outline = self.deduplicate_outline(outline)
        outline.sort(key=lambda x: (x['page'], x['level']))
        
        return outline
    
    def is_heading_like(self, text, level):
        """Check if text looks like a heading"""
        # Check for common heading patterns
        heading_patterns = [
            r'^\d+\.?\s+',  # "1. Introduction" or "1 Introduction"
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # "Title Case Words"
            r'^[A-Z\s]+$',  # "ALL CAPS"
            r'^(Chapter|Section|Part)\s+\d+',  # "Chapter 1", "Section 2"
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        # Check for title case (most words capitalized)
        words = text.split()
        if len(words) >= 2:
            capitalized = sum(1 for word in words if word[0].isupper())
            if capitalized / len(words) >= 0.6:
                return True
        
        return False
    
    def is_likely_body_text(self, text):
        """Check if text is likely body text rather than a heading"""
        # Skip very long paragraphs
        if len(text) > 200:
            return True
        
        # Skip text with lots of lowercase words
        words = text.split()
        if len(words) > 10:
            lowercase_ratio = sum(1 for word in words if word.islower()) / len(words)
            if lowercase_ratio > 0.7:
                return True
        
        # Skip common body text patterns
        body_patterns = [
            r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
            r'\b(this|that|these|those|such|which|what)\b',
            r'[.!?]\s+[A-Z]',  # Multiple sentences
        ]
        
        for pattern in body_patterns:
            if len(re.findall(pattern, text, re.IGNORECASE)) > 2:
                return True
        
        return False
    
    def is_title_like(self, text):
        """Check if text looks like a document title"""
        # Common title indicators
        title_keywords = [
            'introduction', 'overview', 'guide', 'manual', 'handbook',
            'report', 'analysis', 'study', 'research', 'paper'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in title_keywords)
    
    def is_valid_heading(self, text):
        """Validate if text is a proper heading"""
        text = text.strip()
        
        # Check length constraints
        if len(text) < self.min_heading_length or len(text) > self.max_heading_length:
            return False
        
        # Skip pure numbers or special characters
        if text.isdigit() or not any(c.isalpha() for c in text):
            return False
        
        # Skip common non-heading patterns
        skip_patterns = [
            r'^\d+$',  # Pure numbers
            r'^page\s+\d+',  # Page numbers
            r'^figure\s+\d+',  # Figure captions
            r'^table\s+\d+',  # Table captions
            r'^www\.',  # URLs
            r'@',  # Email addresses
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        
        return True
    
    def clean_heading_text(self, text):
        """Clean and normalize heading text"""
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove common prefixes
        text = re.sub(r'^\d+\.?\s*', '', text)  # Remove numbering
        text = re.sub(r'^(Chapter|Section|Part)\s+\d+\.?\s*', '', text, re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove trailing dots and special characters
        text = text.rstrip('.-:')
        
        return text.strip()
    
    def deduplicate_outline(self, outline):
        """Remove duplicate headings"""
        seen = set()
        deduplicated = []
        
        for item in outline:
            key = (item['text'].lower(), item['level'])
            if key not in seen:
                seen.add(key)
                deduplicated.append(item)
        
        return deduplicated
    
    def extract_title_from_doc(self, doc):
        """Extract title from document metadata or first page"""
        # Try metadata first
        metadata = doc.metadata
        if metadata.get('title'):
            return metadata['title'].strip()
        
        # Fall back to text analysis
        if len(doc) > 0:
            first_page = doc[0]
            text_dict = first_page.get_text("dict")
            
            # Look for large text on first page
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            size = span.get("size", 0)
                            if text and len(text) > 5 and size > 14:
                                return self.clean_heading_text(text)
        
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
    
    # Initialize extractor
    extractor = PDFStructureExtractor()
    
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
            
            # Extract structure
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
    logger.info("Starting PDF Structure Extractor - Challenge 1a")
    process_pdfs()
    logger.info("Processing completed")
