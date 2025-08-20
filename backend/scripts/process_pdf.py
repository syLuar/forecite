#!/usr/bin/env python3
"""
PDF Processing Module for Legal Document Analysis

This module consolidates PDF extraction and footnote integration functionality
to provide a single interface for processing legal PDFs with proper footnote handling.

Features:
- Enhanced PDF text extraction using pdfplumber
- Footnote detection and separation
- Header removal and content classification
- Footnote integration into main content
- Clean text formatting for downstream processing

Usage:
    from process_pdf import process_pdf_document

    pages = process_pdf_document(pdf_path)
    for page in pages:
        print(f"Page {page['page']}: {len(page['content'])} chars")
"""

import pdfplumber
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import statistics
import logging


class PDFProcessor:
    """Comprehensive PDF processor for legal documents with footnote integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PDF processor.

        Args:
            config: Optional configuration dictionary with extraction parameters
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for extraction."""
        return {
            "header_region_threshold": 0.12,  # Top 12% of page for headers
            "footnote_region_threshold": 0.8,  # Bottom 20% of page for footnotes
            "font_size_threshold": 0.95,  # Font size ratio for small text detection
            "max_header_length": 80,  # Max characters for header lines
            "max_footnote_length": 120,  # Max characters for standalone footnotes
            "sample_pages_for_analysis": 10,  # Pages to sample for document analysis
        }

    def analyze_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze document structure to understand font patterns and layout.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dict containing document-wide statistics and patterns
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_font_sizes = []
                page_count = len(pdf.pages)
                sample_pages = min(self.config["sample_pages_for_analysis"], page_count)

                for i in range(sample_pages):
                    page = pdf.pages[i]
                    chars = page.chars
                    font_sizes = [
                        char.get("size", 0) for char in chars if char.get("size")
                    ]
                    all_font_sizes.extend(font_sizes)

                if all_font_sizes:
                    main_font_size = statistics.mode(all_font_sizes)
                    font_stats = {
                        "main": main_font_size,
                        "min": min(all_font_sizes),
                        "max": max(all_font_sizes),
                        "median": statistics.median(all_font_sizes),
                    }
                else:
                    font_stats = {"main": 12, "min": 8, "max": 16, "median": 12}

                return {
                    "page_count": page_count,
                    "font_stats": font_stats,
                    "sample_pages": sample_pages,
                }

        except Exception as e:
            self.logger.error(f"Error analyzing document structure: {e}")
            return {
                "page_count": 0,
                "font_stats": {"main": 12, "min": 8, "max": 16, "median": 12},
                "sample_pages": 0,
            }

    def detect_footnote_format(self, pdf_path: str) -> str:
        """
        Detect the footnote format used in the document.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            'numbered' for traditional numbered footnotes, 'bracketed' for [note: X] format
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                # Check for [note: X] references in first few pages
                bracketed_refs = 0
                numbered_refs = 0

                # Sample first 10 pages or all pages if fewer
                sample_pages = min(10, total_pages)
                for i in range(sample_pages):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text:
                        # Count [note: X] references
                        bracketed_matches = re.findall(
                            r"\[note:\s*\d+\]", text, re.IGNORECASE
                        )
                        bracketed_refs += len(bracketed_matches)

                        # Count numbered footnote references (.87, .132, etc.)
                        numbered_matches = re.findall(r"\.(\d+)(?=\s|$|[^\d])", text)
                        numbered_refs += len(numbered_matches)

                # Also check last few pages for end-of-document footnote patterns
                footnote_pages = 0
                for i in range(max(0, total_pages - 5), total_pages):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text and "[note:" in text.lower():
                        footnote_pages += 1

                self.logger.info(
                    f"Footnote format detection: bracketed_refs={bracketed_refs}, numbered_refs={numbered_refs}, footnote_pages={footnote_pages}"
                )

                # Decision logic: if we find [note: X] patterns or dedicated footnote pages, it's bracketed format
                if bracketed_refs > 0 or footnote_pages > 0:
                    return "bracketed"
                else:
                    return "numbered"

        except Exception as e:
            self.logger.error(f"Error detecting footnote format: {e}")
            return "numbered"  # Default to numbered format

    def extract_end_of_document_footnotes(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract footnotes from the end of the document (for bracketed format).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary mapping footnote numbers to their text content
        """
        footnote_map = {}

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                # Check last 5 pages for footnotes
                for i in range(max(0, total_pages - 5), total_pages):
                    page = pdf.pages[i]
                    text = page.extract_text()

                    if text and "[note:" in text.lower():
                        lines = text.split("\n")

                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue

                            # Match [note: X]content pattern
                            match = re.match(
                                r"^\[note:\s*(\d+)\](.+)", line, re.IGNORECASE
                            )
                            if match:
                                footnote_num = match.group(1)
                                footnote_content = match.group(2).strip()

                                # Clean up the footnote content
                                footnote_content = re.sub(r"\s+", " ", footnote_content)
                                footnote_map[footnote_num] = footnote_content

                self.logger.info(
                    f"Extracted {len(footnote_map)} footnotes from end of document"
                )
                return footnote_map

        except Exception as e:
            self.logger.error(f"Error extracting end-of-document footnotes: {e}")
            return {}

    def is_footnote_text(
        self,
        text: str,
        y_position: float,
        page_height: float,
        font_size: float,
        main_font_size: float,
    ) -> bool:
        """Determine if text line is likely a footnote."""
        text = text.strip()

        # Position check
        in_footnote_region = (
            y_position > page_height * self.config["footnote_region_threshold"]
        )

        # Font size check
        is_small_font = font_size < main_font_size * self.config["font_size_threshold"]

        # Footnote patterns for legal documents
        footnote_patterns = [
            r"^\s*\d+\s+",  # Numbered footnotes
            r"^\s*\d+\s*[A-Z]",  # Number + text
            r"^\s*\[note:\s*\d+\]",  # Bracketed footnotes [note: X]
            r"ROA Vol",  # Record of Appeal volumes
            r"GD at \[\d+\]",  # Grounds of Decision references
            r"Appellant\'s case",  # Case references
            r"Respondent\'s case",  # Case references
            r"Version No \d+:",  # Version stamps
            r"^\s*\*",  # Asterisk footnotes
            r"\d{4} \(\d{2}:\d{2} hrs\)",  # Timestamps
            r"pp \d+[–-]\d+",  # Page ranges
            r"para[s]? \d+",  # Paragraph references
        ]

        matches_pattern = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in footnote_patterns
        )

        # Length check
        is_short = len(text) < self.config["max_footnote_length"]

        # Decision logic
        if matches_pattern and (in_footnote_region or is_small_font):
            return True
        if in_footnote_region and is_small_font and is_short:
            return True

        return False

    def is_header_text(
        self,
        text: str,
        y_position: float,
        page_height: float,
        font_size: float,
        main_font_size: float,
    ) -> bool:
        """Determine if text line is likely a header."""
        text = text.strip()

        # Position check
        in_header_region = (
            y_position < page_height * self.config["header_region_threshold"]
        )

        # Header patterns for legal documents - made more specific to avoid false positives
        header_patterns = [
            r"^\d+$",  # Page numbers (standalone)
            r"^COURT OF APPEAL$",  # Court names (standalone)
            r"^SINGAPORE$",  # Jurisdiction (standalone)
            r"^\[\d{4}\] SGCA \d+",  # Case citations (at start)
            r"^Civil Appeal No",  # Appeal numbers (at start)
            r"^Version No \d+:",  # Version stamps (at start)
            r"^(IN THE )?COURT OF",  # Court headers (at start)
            r"^(HIGH COURT|SUPREME COURT)",  # Court names (at start)
        ]

        # Check patterns - most should only match at start or as standalone
        matches_pattern = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in header_patterns
        )

        # Additional check for jurisdiction words only if in header region and short
        if not matches_pattern and in_header_region and len(text) < 20:
            jurisdiction_patterns = [
                r"^SINGAPORE$",  # Only standalone SINGAPORE
                r"COURT OF APPEAL",  # Court names anywhere in short headers
            ]
            matches_pattern = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in jurisdiction_patterns
            )

        # Other header indicators
        is_short_header = (
            len(text) < self.config["max_header_length"] and in_header_region
        )
        is_all_caps = text.isupper() and len(text) > 3 and in_header_region

        return matches_pattern or is_short_header or is_all_caps

    def extract_text_with_metadata(self, page) -> List[Dict[str, Any]]:
        """Extract text lines with position and font metadata."""
        lines = page.extract_text_lines()
        chars = page.chars
        result = []

        for line in lines:
            text = line["text"].strip()
            if not text:
                continue

            # Find characters within this line's boundaries
            line_chars = [
                char
                for char in chars
                if (
                    char["top"] >= line["top"] - 2
                    and char["bottom"] <= line["bottom"] + 2
                )
            ]

            # Calculate average font size for the line
            if line_chars:
                font_sizes = [
                    char.get("size", 0)
                    for char in line_chars
                    if char.get("size", 0) > 0
                ]
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
            else:
                avg_font_size = 0

            result.append(
                {
                    "text": text,
                    "top": line["top"],
                    "bottom": line["bottom"],
                    "font_size": avg_font_size,
                    "char_count": len(line_chars),
                }
            )

        return result

    def classify_content(self, page, doc_stats: Dict[str, Any]) -> Dict[str, List[str]]:
        """Classify page content into headers, body text, and footnotes."""
        lines_metadata = self.extract_text_with_metadata(page)
        page_height = page.height
        main_font_size = doc_stats["font_stats"]["main"]

        headers = []
        body_content = []
        footnotes = []

        for line_meta in lines_metadata:
            text = line_meta["text"]
            y_pos = line_meta["top"]
            font_size = line_meta["font_size"]

            if self.is_header_text(text, y_pos, page_height, font_size, main_font_size):
                headers.append(text)
            elif self.is_footnote_text(
                text, y_pos, page_height, font_size, main_font_size
            ):
                footnotes.append(text)
            else:
                body_content.append(text)

        return {"headers": headers, "body": body_content, "footnotes": footnotes}

    def clean_text(self, text_lines: List[str]) -> str:
        """Clean and format extracted text."""
        if not text_lines:
            return ""

        # Join lines
        text = "\n".join(text_lines)

        # Clean up whitespace
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Reduce multiple newlines
        text = re.sub(r" +", " ", text)  # Reduce multiple spaces
        text = re.sub(r"\n ", "\n", text)  # Remove leading spaces after newlines

        return text.strip()

    def extract_footnote_mappings(self, footnote_text: str) -> Dict[str, str]:
        """
        Extract footnote number to text mappings from footnote content.

        Args:
            footnote_text: Raw footnote content

        Returns:
            Dictionary mapping footnote numbers to their text content
        """
        footnote_map = {}

        if not footnote_text.strip():
            return footnote_map

        # More precise splitting - only split on footnote numbers at start of lines
        # or after newlines, not in the middle of dates/text
        footnote_lines = re.split(
            r"(?=^\d+\s)|(?=\n\d+\s)", footnote_text, flags=re.MULTILINE
        )

        for line in footnote_lines:
            line = line.strip()
            if not line:
                continue

            # Match footnote number at the beginning
            match = re.match(r"^(\d+)\s+(.+)", line, re.DOTALL)
            if match:
                footnote_num = match.group(1)
                footnote_content = match.group(2).strip()
                # Clean up the footnote content - normalize whitespace but preserve structure
                footnote_content = re.sub(r"\s+", " ", footnote_content)
                footnote_map[footnote_num] = footnote_content

        return footnote_map

    def integrate_footnotes_in_content(
        self, content: str, footnote_map: Dict[str, str]
    ) -> str:
        """
        Integrate footnotes into content by replacing footnote references with inline citations.

        Args:
            content: Main content text
            footnote_map: Dictionary mapping footnote numbers to their text

        Returns:
            Content with integrated footnotes
        """
        if not footnote_map:
            return content

        # Find all footnote references in the content
        # Pattern matches: number at end of sentence, after period, etc.
        def replace_footnote_ref(match):
            footnote_num = match.group(1)
            if footnote_num in footnote_map:
                footnote_text = footnote_map[footnote_num]
                # Return the footnote text in parentheses, removing the number prefix if it exists
                if footnote_text.startswith(footnote_num + " "):
                    footnote_text = footnote_text[len(footnote_num) + 1 :]
                return f" ({footnote_text})"
            else:
                # Keep original reference if footnote not found
                return match.group(0)

        # Pattern to match footnote references
        # Matches: .87, Singapore.87, etc. (number after period or at end of word)
        footnote_pattern = r"\.(\d+)(?=\s|$|[^\d])"
        integrated_content = re.sub(footnote_pattern, replace_footnote_ref, content)

        # Also handle footnote references that appear without a preceding period
        # Matches: word87, Singapore87 (number directly after word)
        footnote_pattern2 = r"(\w)(\d+)(?=\s|$|[^\d])"

        def replace_footnote_ref2(match):
            word_char = match.group(1)
            footnote_num = match.group(2)
            if footnote_num in footnote_map:
                footnote_text = footnote_map[footnote_num]
                if footnote_text.startswith(footnote_num + " "):
                    footnote_text = footnote_text[len(footnote_num) + 1 :]
                return f"{word_char} ({footnote_text})"
            else:
                return match.group(0)

        integrated_content = re.sub(
            footnote_pattern2, replace_footnote_ref2, integrated_content
        )

        return integrated_content

    def integrate_bracketed_footnotes_in_content(
        self, content: str, footnote_map: Dict[str, str]
    ) -> str:
        """
        Integrate bracketed footnotes ([note: X]) into content.

        Args:
            content: Main content text
            footnote_map: Dictionary mapping footnote numbers to their text

        Returns:
            Content with integrated footnotes
        """
        if not footnote_map:
            return content

        def replace_footnote_ref(match):
            footnote_num = match.group(1)
            if footnote_num in footnote_map:
                footnote_text = footnote_map[footnote_num]
                return f" ({footnote_text})"
            else:
                # Keep original reference if footnote not found
                return match.group(0)

        # Pattern to match [note: X] references
        footnote_pattern = r"\[note:\s*(\d+)\]"
        integrated_content = re.sub(
            footnote_pattern, replace_footnote_ref, content, flags=re.IGNORECASE
        )

        return integrated_content

    def extract_page(
        self,
        page_num: int,
        pdf_path: str,
        doc_stats: Dict[str, Any],
        footnote_format: str = "numbered",
        end_of_doc_footnotes: Dict[str, str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract content from a single page with footnote integration."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return None

                page = pdf.pages[page_num]
                classified_content = self.classify_content(page, doc_stats)

                # Clean content and footnotes
                content = self.clean_text(classified_content["body"])
                footnotes = self.clean_text(classified_content["footnotes"])

                # Handle footnote integration based on format
                if footnote_format == "bracketed" and end_of_doc_footnotes:
                    # Use end-of-document footnotes for integration
                    integrated_content = self.integrate_bracketed_footnotes_in_content(
                        content, end_of_doc_footnotes
                    )
                    # Count how many footnote references were found on this page
                    footnote_refs = re.findall(
                        r"\[note:\s*(\d+)\]", content, re.IGNORECASE
                    )
                    footnote_count = len(footnote_refs)
                else:
                    # Traditional numbered footnotes - extract from page-level footnotes
                    footnote_map = self.extract_footnote_mappings(footnotes)
                    integrated_content = self.integrate_footnotes_in_content(
                        content, footnote_map
                    )
                    footnote_count = len(footnote_map)

                integrated_content = re.sub(
                    r"(?m)^(\d+)", r"\n\n\1", integrated_content
                )

                return {
                    "page": page_num + 1,
                    "content": integrated_content,
                    "footnotes": footnotes,
                    "original_content": content,
                    "footnote_count": footnote_count,
                }

        except Exception as e:
            self.logger.error(f"Error extracting page {page_num + 1}: {e}")
            return None

    def extract_all_pages(
        self, pdf_path: str, max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Extract content from all pages in the document with footnote integration."""
        self.logger.info(f"Processing PDF: {pdf_path}")
        self.logger.info("Analyzing document structure...")
        doc_stats = self.analyze_document_structure(pdf_path)

        if doc_stats["page_count"] == 0:
            self.logger.error("Failed to analyze document or document is empty")
            return []

        # Detect footnote format
        footnote_format = self.detect_footnote_format(pdf_path)
        self.logger.info(f"Detected footnote format: {footnote_format}")

        # For bracketed format, extract end-of-document footnotes once
        end_of_doc_footnotes = {}
        if footnote_format == "bracketed":
            end_of_doc_footnotes = self.extract_end_of_document_footnotes(pdf_path)

        self.logger.info(
            f"Found {doc_stats['page_count']} pages, main font size: {doc_stats['font_stats']['main']}"
        )

        results = []
        total_pages = doc_stats["page_count"]

        if max_pages:
            total_pages = min(total_pages, max_pages)

        total_footnotes = 0
        for i in range(total_pages):
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{total_pages} pages")

            page_content = self.extract_page(
                i, pdf_path, doc_stats, footnote_format, end_of_doc_footnotes
            )
            if page_content:
                results.append(page_content)
                total_footnotes += page_content["footnote_count"]

        self.logger.info(
            f"Successfully extracted {len(results)} pages with {total_footnotes} total footnotes integrated"
        )
        return results

    def save_to_json(self, pages: List[Dict[str, Any]], output_path: str) -> bool:
        """Save extracted pages to JSON file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pages, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved extraction results to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save to {output_path}: {e}")
            return False


def process_pdf_document(
    pdf_path: str,
    config: Optional[Dict[str, Any]] = None,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    High-level function to process a PDF document with footnote integration.

    Args:
        pdf_path: Path to the PDF file
        config: Optional configuration for processing
        max_pages: Optional limit on number of pages to process

    Returns:
        List of page dictionaries with integrated content
    """
    processor = PDFProcessor(config)
    return processor.extract_all_pages(pdf_path, max_pages)
