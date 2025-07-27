# PDF Structure Extractor - Adobe India Hackathon 2025 Challenge 1a

## Overview

This solution addresses Challenge 1a of the Adobe India Hackathon 2025, focusing on extracting structured outlines from PDF documents. The system processes PDF files to identify document titles and hierarchical headings (H1, H2, H3) with their corresponding page numbers, outputting results in JSON format.

## Solution Architecture

### Multi-Strategy Approach

The solution employs a sophisticated multi-strategy approach for optimal heading detection:

1. **Built-in Table of Contents Analysis**: Prioritizes native PDF structure when available for fastest processing
2. **Advanced Font Analysis**: Analyzes font sizes, styles, and formatting characteristics
3. **Pattern Recognition**: Utilizes language-specific numbering patterns and structural indicators
4. **Context-Aware Classification**: Employs document structure understanding and confidence scoring

### Multilingual Support

The system provides comprehensive multilingual support for bonus scoring:

- **Japanese**: Supports traditional heading structures (章, 節, 項) with proper numbering systems
- **Chinese**: Handles both Traditional and Simplified Chinese formatting patterns
- **Korean**: Recognizes modern Korean document structures (장, 절, 부)
- **English**: Advanced pattern matching for Western document conventions

### Performance Optimizations

- **Parallel Processing**: Utilizes ThreadPoolExecutor for concurrent page processing
- **Smart Caching**: Implements caching for font analysis and pattern matching results
- **Memory Management**: Efficient resource cleanup and garbage collection
- **Pre-compiled Patterns**: All regex patterns compiled at initialization for maximum speed

## Technical Specifications

| Specification | Requirement | Implementation |
|---------------|-------------|----------------|
| Execution Time | ≤ 10 seconds (50-page PDF) | Optimized parallel processing |
| Model Size | ≤ 200MB | No ML models used |
| Network Access | Offline operation only | Zero external dependencies |
| Platform | AMD64 (x86_64) | Docker platform specification |
| Runtime Environment | CPU-only, 8 cores, 16GB RAM | Resource-optimized implementation |

## Dependencies

### Core Libraries
- **PyMuPDF (fitz)**: Fast PDF processing and text extraction
- **Python Standard Library**: json, pathlib, re, threading, concurrent.futures

### System Requirements
- Python 3.10
- Linux AMD64 architecture
- UTF-8 encoding support

## Installation and Setup

### Prerequisites
- Docker installed and running
- Input PDF files prepared
- Sufficient system resources (8 CPU cores, 16GB RAM recommended)

### Build Instructions

1. **Prepare project files**
* Clone the repository using `git clone https://github.com/your-username/your-r
* Install required dependencies using `pip install -r requirements.txt`
* Build the Docker image using `docker build -t bonus-scoring .`
* Run the Docker container using `docker run -it --rm -v $(pwd):/app
* Verify the system's performance and accuracy using the provided test cases