# PDF Structure Extractor - Adobe India Hackathon 2025 Challenge 1a

## Overview
This solution extracts structured outlines (title and headings H1, H2, H3) from PDF documents and outputs them in JSON format. The solution is designed to meet all challenge requirements including performance constraints and offline operation.

## Approach

### Multi-Strategy Heading Detection
1. **Built-in TOC Analysis**: First attempts to use PDF's native table of contents if available
2. **Font-Based Analysis**: Analyzes font sizes, styles, and formatting to identify headings
3. **Pattern Recognition**: Uses regex patterns to identify common heading structures
4. **Context-Aware Classification**: Considers document structure and text patterns

### Key Features
- **Performance Optimized**: Processes 50-page PDFs in under 10 seconds
- **Robust Error Handling**: Gracefully handles malformed or complex PDFs
- **Multi-Language Support**: Works with various character encodings
- **Memory Efficient**: Optimized for 16GB RAM constraint
- **Offline Operation**: No internet dependencies

### Heading Detection Logic
- **Title Extraction**: Identifies document title using font size, position, and metadata
- **H1 Detection**: Large fonts, bold formatting, common heading patterns
- **H2 Detection**: Medium fonts with emphasis, section-like structure
- **H3 Detection**: Smaller but distinct fonts, subsection patterns

## Libraries Used
- **PyMuPDF (fitz)**: Fast PDF processing and text extraction
- **Standard Python Libraries**: json, pathlib, re, collections, logging

## Technical Specifications
- **Platform**: AMD64 (x86_64) compatible
- **Runtime**: CPU-only, optimized for 8 CPUs
- **Memory**: Designed for 16GB RAM systems
- **Model Size**: No ML models used (under 200MB constraint)
- **Execution Time**: < 10 seconds for 50-page PDFs

## Build Instructions

### Build the Docker Image
