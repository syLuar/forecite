#!/usr/bin/env python3
"""
Singapore Judiciary PDF Scraper

This script scrapes PDF case documents from the Singapore Judiciary website
(https://www.judiciary.gov.sg/judgments/judgments-case-summaries) and saves them
to the data/raw_docs/cases directory.

The script:
1. Fetches the list of judgments from the paginated judgment portal
2. Extracts case details and legal tags/catchwords directly from the list view
3. Downloads PDF files for each case
4. Organizes files by year and saves with descriptive names
5. Creates comprehensive metadata JSON file with legal tags for each case
6. Provides progress tracking and resume capability

The metadata includes:
- Case ID, title, citation, court, decision date
- Legal tags/catchwords (e.g., "Companies — Directors — Duties")
- Parties involved, case numbers
- File paths and download information

Usage:
    python scrape_singapore_cases.py [--year YEAR] [--max-pages N] [--resume]
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class SingaporeJudiciaryScaper:
    """Scraper for Singapore Judiciary PDF documents."""

    def __init__(self, output_dir: str = None, delay: float = 1.0):
        """
        Initialize the scraper.

        Args:
            output_dir: Directory to save PDFs (default: data/raw_docs/cases)
            delay: Delay between requests in seconds (default: 1.0)
        """
        self.base_url = "https://www.elitigation.sg"
        self.search_url = f"{self.base_url}/gd/Home/Index"
        self.delay = delay

        # Set up output directory
        if output_dir is None:
            output_dir = (
                Path(__file__).parent.parent.parent / "data" / "raw_docs" / "cases"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("scraping.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        # Progress tracking
        self.progress_file = self.output_dir / "scraping_progress.json"
        self.metadata_file = self.output_dir / "cases_metadata.json"
        self.downloaded_cases: Set[str] = set()
        self.failed_cases: Set[str] = set()
        self.cases_metadata: Dict[str, Dict] = {}
        self.load_progress()
        self.load_metadata()

    def load_progress(self):
        """Load previous scraping progress."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                    self.downloaded_cases = set(data.get("downloaded", []))
                    self.failed_cases = set(data.get("failed", []))
                self.logger.info(
                    f"Loaded progress: {len(self.downloaded_cases)} downloaded, "
                    f"{len(self.failed_cases)} failed"
                )
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}")

    def save_progress(self):
        """Save current scraping progress."""
        try:
            data = {
                "downloaded": list(self.downloaded_cases),
                "failed": list(self.failed_cases),
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.progress_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save progress: {e}")

    def load_metadata(self):
        """Load existing metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    self.cases_metadata = json.load(f)
                self.logger.info(
                    f"Loaded metadata for {len(self.cases_metadata)} cases"
                )
            except Exception as e:
                self.logger.warning(f"Could not load metadata file: {e}")

    def save_metadata(self):
        """Save cases metadata."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.cases_metadata, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved metadata for {len(self.cases_metadata)} cases")
        except Exception as e:
            self.logger.error(f"Could not save metadata: {e}")

    def extract_case_metadata(self, case_url: str, case_id: str) -> Dict:
        """
        Extract detailed metadata from a case page including tags.

        Args:
            case_url: URL of the individual case page
            case_id: Case ID

        Returns:
            Dictionary with case metadata including tags
        """
        metadata = {
            "case_id": case_id,
            "url": case_url,
            "tags": [],
            "extracted_at": datetime.now().isoformat(),
            "title": "",
            "decision_date": "",
            "case_number": "",
            "court": "",
            "judge": "",
            "parties": [],
            "summary": "",
        }

        try:
            self.logger.debug(f"Extracting metadata from {case_url}")
            response = self.session.get(case_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title
            title_elem = soup.find("h1") or soup.find("title")
            if title_elem:
                metadata["title"] = title_elem.get_text(strip=True)

            # Extract tags - these appear in various formats
            # Look for bracketed content that contains legal topics
            page_text = soup.get_text()

            # Pattern 1: Tags in brackets like [Companies — Directors — Duties]
            bracket_pattern = r"\[([^\]]+(?:—[^\]]+)*)\]"
            bracket_matches = re.findall(bracket_pattern, page_text)

            for match in bracket_matches:
                # Split on em dash or regular dash and clean up
                tags = [tag.strip() for tag in re.split(r"[—–-]", match)]
                # Filter out non-tag content (like years, case numbers)
                legal_tags = []
                for tag in tags:
                    # Skip if it's just a year or case number pattern
                    if not re.match(r"^\d{4}$", tag) and not re.match(
                        r"^[A-Z]+\s*\d+$", tag
                    ):
                        if len(tag) > 2:  # Reasonable tag length
                            legal_tags.append(tag)

                if legal_tags:
                    metadata["tags"].extend(legal_tags)

            # Pattern 2: Look for section headings that might contain topic areas
            headings = soup.find_all(["h2", "h3", "h4", "strong", "b"])
            for heading in headings:
                heading_text = heading.get_text(strip=True)
                # Look for legal topic patterns
                if any(
                    keyword in heading_text.lower()
                    for keyword in [
                        "law",
                        "contract",
                        "tort",
                        "evidence",
                        "procedure",
                        "damages",
                        "liability",
                        "trust",
                        "company",
                        "insolvency",
                        "property",
                    ]
                ):
                    if heading_text not in metadata["tags"] and len(heading_text) > 5:
                        metadata["tags"].append(heading_text)

            # Extract other metadata elements
            # Look for decision date
            date_patterns = [
                r"Decision Date:\s*(\d{1,2}\s+\w+\s+\d{4})",
                r"Date of Decision:\s*(\d{1,2}\s+\w+\s+\d{4})",
                r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, page_text, re.IGNORECASE)
                if date_match:
                    metadata["decision_date"] = date_match.group(1)
                    break

            # Extract case number
            case_number_patterns = [
                r"([A-Z]+(?:\s*[A-Z]*)*\s*[\/\\]\s*\d+[\/\\]\d{4})",
                r"(\[\d{4}\]\s*[A-Z]+\s*\d+)",
                r"([A-Z]+\s*\d+\s*of\s*\d{4})",
            ]

            for pattern in case_number_patterns:
                case_match = re.search(pattern, page_text)
                if case_match:
                    metadata["case_number"] = case_match.group(1).strip()
                    break

            # Extract court information
            court_patterns = [
                r"(Supreme Court|High Court|Court of Appeal|District Court|Magistrate)",
                r"(SGHC|SGCA|SGDC|SGMC)",
            ]

            for pattern in court_patterns:
                court_match = re.search(pattern, page_text, re.IGNORECASE)
                if court_match:
                    metadata["court"] = court_match.group(1)
                    break

            # Look for judge information
            judge_patterns = [
                r"(?:Before|Coram):\s*([^.\n]+(?:JC?|CJ))",
                r"([A-Z][a-z]+\s+[A-Z][a-z]+\s+JC?)",
                r"Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            ]

            for pattern in judge_patterns:
                judge_match = re.search(pattern, page_text)
                if judge_match:
                    metadata["judge"] = judge_match.group(1).strip()
                    break

            # Extract parties (plaintiff vs defendant)
            party_patterns = [
                r"([A-Z][A-Z\s&.,]+)\s+v\.?\s+([A-Z][A-Z\s&.,]+)",
                r"Between:\s*([^;\n]+)\s+and\s+([^;\n]+)",
            ]

            for pattern in party_patterns:
                party_match = re.search(pattern, page_text)
                if party_match:
                    metadata["parties"] = [
                        party_match.group(1).strip(),
                        party_match.group(2).strip(),
                    ]
                    break

            # Clean up tags - remove duplicates and very short ones
            unique_tags = []
            seen_tags = set()
            for tag in metadata["tags"]:
                tag_clean = tag.strip()
                tag_lower = tag_clean.lower()
                if tag_clean and len(tag_clean) > 2 and tag_lower not in seen_tags:
                    unique_tags.append(tag_clean)
                    seen_tags.add(tag_lower)

            metadata["tags"] = unique_tags[:20]  # Limit to 20 most relevant tags

            # Add delay to be respectful
            time.sleep(self.delay * 0.5)

            return metadata

        except Exception as e:
            self.logger.warning(f"Could not extract metadata for {case_id}: {e}")
            return metadata

    def get_judgments_page(self, page: int = 1, year: str = "All") -> Optional[Dict]:
        """
        Fetch a page of judgments from the search interface.

        Args:
            page: Page number to fetch
            year: Year filter ("All" or specific year like "2024")

        Returns:
            Dictionary with judgment data or None if failed
        """
        params = {
            "Filter": "SUPCT",  # Supreme Court filter
            "YearOfDecision": year,
            "SortBy": "DateOfDecision",
            "CurrentPage": page,
            "SortAscending": "False",
            "PageSize": "0",  # Use default page size
            "Verbose": "False",
            "SearchQueryTime": "0",
            "SearchTotalHits": "0",
            "SearchMode": "True",
            "SpanMultiplePages": "False",
        }

        try:
            self.logger.info(f"Fetching page {page} for year {year}")
            response = self.session.get(self.search_url, params=params, timeout=30)
            response.raise_for_status()

            # Parse the response
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract judgment data
            judgments = self.parse_judgments_page(soup)

            # Extract pagination info
            pagination_info = self.parse_pagination(soup)

            return {
                "judgments": judgments,
                "pagination": pagination_info,
                "page": page,
                "year": year,
            }

        except Exception as e:
            self.logger.error(f"Failed to fetch page {page}: {e}")
            return None

    def parse_judgments_page(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Parse judgment entries from a search results page.

        Args:
            soup: BeautifulSoup object of the page

        Returns:
            List of judgment dictionaries
        """
        judgments = []

        # Look for judgment card containers
        judgment_cards = soup.find_all("div", class_="card")

        for card in judgment_cards:
            try:
                # Find the main case link
                case_link = card.find("a", href=re.compile(r"/gd/s/\d{4}_\w+"))
                if not case_link:
                    continue

                href = case_link.get("href", "")

                # Extract case ID from URL
                case_match = re.search(r"/gd/s/(\d{4}_\w+\d+)", href)
                if not case_match:
                    continue

                case_id = case_match.group(1)

                # Extract case title
                title = case_link.get_text(strip=True)
                # Clean up HTML entities
                title = (
                    title.replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                )

                # Extract tags/catchwords from the catchword container
                tags = []
                catchword_container = card.find("div", class_="gd-catchword-container")
                if catchword_container:
                    catchword_links = catchword_container.find_all("a", class_="gd-cw")
                    for cw_link in catchword_links:
                        tag_text = cw_link.get_text(strip=True)
                        # Remove brackets and clean up whitespace, but keep the full tag intact
                        tag_text = tag_text.strip("[]").strip()
                        if tag_text:
                            # Clean up any extra whitespace but preserve the em dashes and structure
                            tag_text = re.sub(r"\s+", " ", tag_text)
                            tags.append(tag_text)

                # Extract additional metadata from the card body
                card_body = card.find("div", class_="gd-card-body")
                additional_info = {}

                if card_body:
                    card_text = card_body.get_text()

                    # Extract citation
                    citation_match = re.search(
                        r"\[(\d{4})\]\s*([A-Z]+)\s*(\d+)", card_text
                    )
                    if citation_match:
                        additional_info["citation"] = (
                            f"[{citation_match.group(1)}] {citation_match.group(2)} {citation_match.group(3)}"
                        )
                        additional_info["year"] = citation_match.group(1)
                        additional_info["court"] = citation_match.group(2)

                    # Extract decision date
                    date_match = re.search(
                        r"Decision Date:\s*(\d{1,2}\s+\w+\s+\d{4})", card_text
                    )
                    if date_match:
                        additional_info["decision_date"] = date_match.group(1)

                    # Extract case number
                    case_num_match = re.search(
                        r"([A-Z]+/[A-Z]*\s*\d+/\d{4})", card_text
                    )
                    if case_num_match:
                        additional_info["case_number"] = case_num_match.group(1)

                # Extract parties from title
                parties = []
                if " v " in title:
                    party_parts = title.split(" v ")
                    if len(party_parts) >= 2:
                        parties = [party_parts[0].strip(), party_parts[1].strip()]
                elif " V " in title:
                    party_parts = title.split(" V ")
                    if len(party_parts) >= 2:
                        parties = [party_parts[0].strip(), party_parts[1].strip()]

                # Create judgment entry with metadata
                judgment = {
                    "case_id": case_id,
                    "title": title,
                    "url": urljoin(self.base_url, href),
                    "pdf_url": f"{self.base_url}/gd/gd/{case_id}/pdf",
                    "tags": tags,
                    "parties": parties,
                    "extracted_at": datetime.now().isoformat(),
                    **additional_info,
                }

                judgments.append(judgment)

            except Exception as e:
                self.logger.warning(f"Could not parse judgment from card: {e}")
                continue

        # Remove duplicates based on case_id
        seen_cases = set()
        unique_judgments = []
        for judgment in judgments:
            if judgment["case_id"] not in seen_cases:
                unique_judgments.append(judgment)
                seen_cases.add(judgment["case_id"])

        self.logger.info(f"Found {len(unique_judgments)} unique judgments on this page")
        return unique_judgments

    def parse_pagination(self, soup: BeautifulSoup) -> Dict:
        """
        Parse pagination information from the page.

        Args:
            soup: BeautifulSoup object of the page

        Returns:
            Dictionary with pagination info
        """
        pagination_info = {
            "current_page": 1,
            "total_pages": 1,
            "has_next": False,
            "total_results": 0,
        }

        try:
            # Look for total results count
            total_text = soup.find(text=re.compile(r"Total Judgment\(s\) Found"))
            if total_text:
                total_match = re.search(r"(\d+)", total_text)
                if total_match:
                    pagination_info["total_results"] = int(total_match.group(1))

            # Look for pagination links
            pagination_links = soup.find_all("a", href=re.compile(r"CurrentPage=\d+"))
            if pagination_links:
                page_numbers = []
                for link in pagination_links:
                    page_match = re.search(r"CurrentPage=(\d+)", link.get("href", ""))
                    if page_match:
                        page_numbers.append(int(page_match.group(1)))

                if page_numbers:
                    pagination_info["total_pages"] = max(page_numbers)
                    pagination_info["has_next"] = True

            # Look for "Last" link to get total pages
            last_link = soup.find("a", text=re.compile(r"Last", re.I))
            if last_link:
                href = last_link.get("href", "")
                last_page_match = re.search(r"CurrentPage=(\d+)", href)
                if last_page_match:
                    pagination_info["total_pages"] = int(last_page_match.group(1))

        except Exception as e:
            self.logger.warning(f"Could not parse pagination info: {e}")

        return pagination_info

    def download_pdf(self, judgment: Dict) -> bool:
        """
        Download a PDF for a specific judgment and store metadata.

        Args:
            judgment: Judgment dictionary with case info and metadata

        Returns:
            True if successful, False otherwise
        """
        case_id = judgment["case_id"]

        # Always store/update metadata regardless of download status
        self.logger.debug(f"Storing/updating metadata for {case_id}")

        # Create comprehensive metadata from judgment data
        metadata = {
            "case_id": case_id,
            "url": judgment["url"],
            "pdf_url": judgment["pdf_url"],
            "title": judgment.get("title", ""),
            "tags": judgment.get("tags", []),
            "decision_date": judgment.get("decision_date", ""),
            "case_number": judgment.get("case_number", ""),
            "citation": judgment.get("citation", ""),
            "court": judgment.get("court", ""),
            "year": judgment.get("year", ""),
            "parties": judgment.get("parties", []),
            "extracted_at": judgment.get("extracted_at", datetime.now().isoformat()),
            "extraction_method": "list_view",
        }

        # Clean and deduplicate tags
        unique_tags = []
        seen_tags = set()
        for tag in metadata["tags"]:
            tag_clean = tag.strip()
            tag_lower = tag_clean.lower()
            if tag_clean and len(tag_clean) > 2 and tag_lower not in seen_tags:
                unique_tags.append(tag_clean)
                seen_tags.add(tag_lower)

        metadata["tags"] = unique_tags

        # Always update metadata (don't check if already exists)
        self.cases_metadata[case_id] = metadata

        # Skip PDF download if already downloaded
        if case_id in self.downloaded_cases:
            self.logger.debug(
                f"Skipping PDF download for {case_id} - already downloaded"
            )
            return True

        # Skip PDF download if previously failed
        if case_id in self.failed_cases:
            self.logger.debug(
                f"Skipping PDF download for {case_id} - previously failed"
            )
            return False

        try:
            # Determine year for organization
            year_match = re.match(r"(\d{4})_", case_id)
            year = (
                year_match.group(1) if year_match else judgment.get("year", "unknown")
            )

            # Create year directory
            year_dir = self.output_dir / year
            year_dir.mkdir(exist_ok=True)

            # Generate filename
            title_clean = re.sub(r"[^\w\s-]", "", judgment.get("title", ""))
            title_clean = re.sub(r"\s+", "_", title_clean.strip())[:100]  # Limit length

            filename = f"{case_id}"
            if title_clean:
                filename += f"_{title_clean}"
            filename += ".pdf"

            file_path = year_dir / filename

            # Update metadata with file path
            self.cases_metadata[case_id]["file_path"] = str(file_path)
            self.cases_metadata[case_id]["filename"] = filename

            # Skip PDF download if file already exists
            if file_path.exists():
                self.logger.info(f"File already exists: {filename}")
                self.downloaded_cases.add(case_id)
                return True

            # Download PDF
            self.logger.info(
                f"Downloading {case_id}: {judgment.get('title', 'Unknown title')}"
            )

            response = self.session.get(judgment["pdf_url"], timeout=60)
            response.raise_for_status()

            # Check if response is actually a PDF
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
                self.logger.warning(
                    f"Response for {case_id} doesn't appear to be a PDF"
                )
                return False

            # Save file
            with open(file_path, "wb") as f:
                f.write(response.content)

            # Update metadata with file info
            self.cases_metadata[case_id].update(
                {
                    "file_size": len(response.content),
                    "downloaded_at": datetime.now().isoformat(),
                    "content_type": content_type,
                }
            )

            self.logger.info(
                f"Successfully downloaded: {filename} ({len(response.content)} bytes)"
            )
            self.downloaded_cases.add(case_id)

            # Add delay between downloads
            time.sleep(self.delay)

            return True

        except Exception as e:
            self.logger.error(f"Failed to download {case_id}: {e}")
            self.failed_cases.add(case_id)
            return False

    def scrape_all(
        self, year: str = "All", max_pages: Optional[int] = None, start_page: int = 1
    ) -> Dict:
        """
        Scrape all judgments for a given year.

        Args:
            year: Year to scrape ("All" or specific year)
            max_pages: Maximum number of pages to scrape
            start_page: Page to start from

        Returns:
            Summary statistics
        """
        self.logger.info(
            f"Starting scrape for year {year}, starting from page {start_page}"
        )

        total_downloaded = 0
        total_failed = 0
        pages_processed = 0

        current_page = start_page

        while True:
            # Fetch page
            page_data = self.get_judgments_page(current_page, year)

            if not page_data or not page_data["judgments"]:
                self.logger.info(f"No more judgments found on page {current_page}")
                break

            pages_processed += 1

            # Download PDFs for this page
            page_downloaded = 0
            page_failed = 0

            for judgment in page_data["judgments"]:
                if self.download_pdf(judgment):
                    page_downloaded += 1
                else:
                    page_failed += 1

            total_downloaded += page_downloaded
            total_failed += page_failed

            self.logger.info(
                f"Page {current_page}: {page_downloaded} downloaded, "
                f"{page_failed} failed"
            )

            # Save progress periodically
            if pages_processed % 5 == 0:
                self.save_progress()
                self.save_metadata()

            # Check if we should continue
            pagination = page_data.get("pagination", {})
            if not pagination.get("has_next", False):
                self.logger.info("Reached last page")
                break

            if max_pages and pages_processed >= max_pages:
                self.logger.info(f"Reached maximum pages limit: {max_pages}")
                break

            current_page += 1

            # Add delay between pages
            time.sleep(self.delay)

        # Final progress save
        self.save_progress()
        self.save_metadata()

        summary = {
            "pages_processed": pages_processed,
            "total_downloaded": total_downloaded,
            "total_failed": total_failed,
            "year": year,
            "final_page": current_page - 1,
            "metadata_entries": len(self.cases_metadata),
        }

        self.logger.info(f"Scraping complete: {summary}")
        return summary


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Scrape Singapore Judiciary PDF cases")
    parser.add_argument("--year", default="All", help="Year to scrape (default: All)")
    parser.add_argument(
        "--max-pages", type=int, help="Maximum number of pages to scrape"
    )
    parser.add_argument(
        "--start-page", type=int, default=1, help="Page to start from (default: 1)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/raw_docs/cases",
        help="Output directory for PDFs",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume previous scraping session"
    )

    args = parser.parse_args()

    # Initialize scraper
    scraper = SingaporeJudiciaryScaper(output_dir=args.output_dir, delay=args.delay)

    # Display resume info if requested
    if args.resume:
        print(f"Resuming scraping...")
        print(f"Previously downloaded: {len(scraper.downloaded_cases)} cases")
        print(f"Previously failed: {len(scraper.failed_cases)} cases")
        print(f"Metadata entries: {len(scraper.cases_metadata)} cases")

    # Start scraping
    try:
        summary = scraper.scrape_all(
            year=args.year, max_pages=args.max_pages, start_page=args.start_page
        )

        print("\nScraping Summary:")
        print(f"  Year: {summary['year']}")
        print(f"  Pages processed: {summary['pages_processed']}")
        print(f"  Total downloaded: {summary['total_downloaded']}")
        print(f"  Total failed: {summary['total_failed']}")
        print(f"  Final page: {summary['final_page']}")
        print(f"  Metadata entries: {summary['metadata_entries']}")

    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        scraper.save_progress()
        scraper.save_metadata()
    except Exception as e:
        print(f"Scraping failed: {e}")
        scraper.save_progress()
        scraper.save_metadata()
        raise


if __name__ == "__main__":
    main()
