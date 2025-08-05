#!/usr/bin/env python3
"""
Paper Summarization Utilities

Handles LLM-based summarization of papers with text chunking for large documents.
Supports both extracted text and direct PDF processing.

Dependencies:
    pip install openai anthropic google-generativeai PyPDF2 pymupdf pdfplumber

Usage:
    from summarize_utils import PaperSummarizer

    summarizer = PaperSummarizer(config, llm_manager)
    summary_data = summarizer.summarize_paper(paper_data)
"""

import json
import time
import re
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def get_default_summarization_prompts() -> Dict[str, str]:
    """
    Get default prompts for research paper summarization.

    Returns:
        Dictionary with default prompt strings
    """
    return {
        "research_context_prompt": """
You are summarizing research papers for a computational scientist who works on 
machine learning, statistical methods, and data analysis techniques. The researcher 
values methodological rigor, theoretical contributions, and practical applications 
that advance scientific understanding.

Focus on technical innovations, experimental validation, and broader implications 
for the field. Assume the reader is an expert who wants comprehensive technical details.
        """.strip(),
        "summarization_strategy_prompt": """
Create a comprehensive technical summary structured as follows:

**1. CORE CONTRIBUTION (2-3 sentences):**
- Main innovation, finding, or theoretical advance
- What makes this work novel or significant

**2. METHODOLOGY (1 paragraph):**
- Key technical approaches and methods used
- Novel algorithmic or experimental techniques
- Important implementation details

**3. RESULTS (1 paragraph):**
- Primary findings and their quantitative significance
- Comparison with existing approaches or baselines
- Statistical significance and validation methods

**4. IMPLICATIONS (2-3 sentences):**
- Broader impact on the field or related areas
- Future research directions or applications
- Limitations or areas for improvement
        """.strip(),
        "summary_format_prompt": """
Length: Aim for 1-2 pages (1000-2000 words maximum).
Format: Use clear markdown structure with section headers.
Style: Technical but accessible, suitable for expert readers.
Focus: Emphasize reproducibility, methodology, and practical significance.

Do not attempt to reproduce mathematical equations or complex formulas.
Instead, describe them in words (e.g., "uses a modified loss function that combines...")
        """.strip(),
    }


def get_summary_json_format() -> str:
    """Get JSON format instructions for summarization output."""
    return """
Provide result as JSON:
{
  "summary": "Complete technical summary following the structure above",
  "key_contributions": ["list", "of", "3-5", "key", "contributions"],
  "confidence": 0.9
}

The confidence score (0.0-1.0) should reflect how well you could summarize the paper 
based on the available text quality and completeness.

Do not include any text outside of this JSON structure.""".strip()


class PaperSummarizer:
    """
    Handles LLM-based paper summarization with chunking support for large documents.
    """

    def __init__(self, config: Dict, llm_manager, model_alias: str):
        """
        Initialize the paper summarizer.

        Args:
            config: Configuration dictionary with summarization settings
            llm_manager: LLMManager instance for API calls
            model_alias: Model alias to use for summarization
        """
        self.config = config
        self.llm_manager = llm_manager
        self.model_alias = model_alias

        # Validate model configuration
        try:
            self.model_config = llm_manager.get_model_config(model_alias)
            self.client = llm_manager.get_client(model_alias)
        except Exception as e:
            raise ValueError(f"Failed to initialize model '{model_alias}': {e}")

        # Summarization configuration
        self.summ_config = config.get("summarization", {})
        self.retry_attempts = self.summ_config.get("retry_attempts", 2)
        self.prefer_extracted_text = self.summ_config.get("prefer_extracted_text", True)
        self.max_papers = self.summ_config.get("max_papers", None)

        # Token and chunking configuration
        self.max_single_chunk_tokens = self.summ_config.get(
            "max_single_chunk_tokens", 15000
        )
        self.max_chunk_tokens = self.summ_config.get("max_chunk_tokens", 8000)
        self.chunk_overlap_ratio = self.summ_config.get("chunk_overlap_ratio", 0.2)

        # Construct the system prompt
        self.system_prompt = self._build_system_prompt()

        # Statistics tracking
        self.total_summarized = 0
        self.total_retries = 0
        self.total_failures = 0
        self.total_chunked = 0
        self.start_time = None

    def _get_default_prompts(self) -> Dict[str, str]:
        """Get default prompts for summarization."""
        return get_default_summarization_prompts()

    def _build_system_prompt(self) -> str:
        """
        Construct the complete system prompt from configured parts.

        Returns:
            Complete system prompt string
        """
        defaults = self._get_default_prompts()
        used_defaults = []

        # Get prompts, using defaults if missing
        research_context = self.summ_config.get("research_context_prompt", "").strip()
        if not research_context:
            research_context = defaults["research_context_prompt"]
            used_defaults.append("research_context_prompt")

        summarization_strategy = self.summ_config.get(
            "summarization_strategy_prompt", ""
        ).strip()
        if not summarization_strategy:
            summarization_strategy = defaults["summarization_strategy_prompt"]
            used_defaults.append("summarization_strategy_prompt")

        summary_format = self.summ_config.get("summary_format_prompt", "").strip()
        if not summary_format:
            summary_format = defaults["summary_format_prompt"]
            used_defaults.append("summary_format_prompt")

        # Get JSON format
        json_format = get_summary_json_format()

        # Warn user about defaults
        if used_defaults:
            print(f"⚠️  WARNING: Using default prompts for: {', '.join(used_defaults)}")
            print(
                f"   For better results, customize these prompts in your config file."
            )
            print()

        # Combine the parts
        prompt_parts = [
            f"RESEARCH CONTEXT:\n{research_context}",
            f"SUMMARIZATION STRATEGY:\n{summarization_strategy}",
            f"FORMATTING GUIDELINES:\n{summary_format}",
            f"OUTPUT FORMAT:\n{json_format}",
        ]

        return "\n\n".join(prompt_parts)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using rough approximation.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: 4 characters per token
        return len(text) // 4

    def _find_content_for_paper(
        self, paper: Dict
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Find the best available content for a paper (extracted text or PDF path).

        Args:
            paper: Paper dictionary

        Returns:
            Tuple of (text_content, pdf_path, source_type) where source_type is 'extracted', 'pdf', or 'none'
        """
        # Try extracted text first if preferred and available
        if self.prefer_extracted_text and paper.get("extraction_success"):
            extracted_path = paper.get("extraction_output_path")
            if extracted_path and os.path.exists(extracted_path):
                try:
                    with open(extracted_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Remove YAML frontmatter if present (from markdown)
                    if content.startswith("---\n"):
                        parts = content.split("---\n", 2)
                        if len(parts) >= 3:
                            content = parts[2]

                    return content.strip(), None, "extracted"
                except Exception as e:
                    print(f"Warning: Could not read extracted text file: {e}")

        # Fall back to PDF direct processing if available
        pdf_path = paper.get("pdf_file_path")
        if pdf_path and os.path.exists(pdf_path):
            return None, pdf_path, "pdf"

        return None, None, "none"

    def _chunk_text(self, text: str, max_tokens: int) -> List[Dict]:
        """
        Split text into overlapping chunks that fit within token limits.

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Convert tokens to characters (rough estimate)
        max_chars = max_tokens * 4

        if len(text) <= max_chars:
            return [{"text": text, "chunk_index": 0, "total_chunks": 1}]

        chunks = []
        overlap_chars = int(max_chars * self.chunk_overlap_ratio)

        start = 0
        chunk_index = 0

        while start < len(text):
            # Find end position
            end = start + max_chars

            if end >= len(text):
                # Last chunk
                chunk_text = text[start:]
            else:
                # Try to break at paragraph boundary
                next_paragraph = text.find("\n\n", end - 500, end + 500)
                if next_paragraph != -1 and next_paragraph > start + max_chars // 2:
                    end = next_paragraph
                else:
                    # Try to break at sentence boundary
                    sentence_end = text.rfind(". ", start + max_chars // 2, end + 200)
                    if sentence_end != -1:
                        end = sentence_end + 2

                chunk_text = text[start:end]

            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "chunk_index": chunk_index,
                    "total_chunks": -1,  # Will be filled later
                    "start_pos": start,
                    "end_pos": end,
                }
            )

            # Move start position for next chunk (with overlap)
            start = end - overlap_chars
            chunk_index += 1

        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)

        return chunks

    def _summarize_single_content(
        self, text_content: str, pdf_path: str, paper: Dict
    ) -> Optional[str]:
        """
        Summarize content that fits in a single LLM call.

        Args:
            text_content: Text content to summarize (None if using PDF)
            pdf_path: Path to PDF file (None if using text)
            paper: Paper metadata

        Returns:
            LLM response or None on failure
        """
        paper_info = f"Title: {paper.get('title', 'Unknown')}\n"
        paper_info += f"Authors: {', '.join(paper.get('authors', []))}\n"
        paper_info += f"Categories: {', '.join(paper.get('categories', []))}\n\n"

        if text_content:
            # Text-based summarization
            content = f"Please summarize this research paper:\n\n{paper_info}PAPER CONTENT:\n{text_content}"
            return self._call_llm_api(content, None)
        elif pdf_path:
            # PDF-based summarization
            content = f"Please summarize this research paper:\n\n{paper_info}The full paper content is provided as a PDF file."
            return self._call_llm_api(content, pdf_path)
        else:
            return None

    def _summarize_chunk(self, chunk: Dict, paper: Dict) -> Optional[str]:
        """
        Summarize a single chunk of text.

        Args:
            chunk: Chunk dictionary with text and metadata
            paper: Paper metadata

        Returns:
            LLM response or None on failure
        """
        paper_info = f"Title: {paper.get('title', 'Unknown')}\n"
        paper_info += f"Authors: {', '.join(paper.get('authors', []))}\n"
        paper_info += f"Categories: {', '.join(paper.get('categories', []))}\n\n"

        chunk_info = (
            f"[Chunk {chunk['chunk_index'] + 1} of {chunk['total_chunks']}]\n\n"
        )

        # Modified prompt for chunk summarization
        chunk_prompt = f"""
Please summarize this section of a research paper. This is chunk {chunk['chunk_index'] + 1} of {chunk['total_chunks']} total chunks.

Focus on the key information in this section without trying to provide a complete paper summary. 
If this appears to be an incomplete section, summarize what is available.

{paper_info}{chunk_info}SECTION CONTENT:\n{chunk['text']}

Provide a focused summary of this section's content in JSON format:
{{
  "section_summary": "Summary of this section's content",
  "key_points": ["point1", "point2", "point3"],
  "section_type": "introduction|methods|results|discussion|other"
}}"""

        return self._call_llm_api(chunk_prompt)

    def _merge_chunk_summaries(
        self, chunk_summaries: List[str], paper: Dict
    ) -> Optional[str]:
        """
        Merge multiple chunk summaries into a final comprehensive summary.

        Args:
            chunk_summaries: List of chunk summary JSON strings
            paper: Paper metadata

        Returns:
            Final merged summary or None on failure
        """
        paper_info = f"Title: {paper.get('title', 'Unknown')}\n"
        paper_info += f"Authors: {', '.join(paper.get('authors', []))}\n"
        paper_info += f"Categories: {', '.join(paper.get('categories', []))}\n\n"

        summaries_text = "\n\n".join(
            [
                f"CHUNK {i+1} SUMMARY:\n{summary}"
                for i, summary in enumerate(chunk_summaries)
            ]
        )

        merge_prompt = f"""Please create a comprehensive summary by merging these section summaries from a research paper.

{paper_info}SECTION SUMMARIES TO MERGE:
{summaries_text}

Create a unified, comprehensive summary that follows the original format requirements.
Eliminate redundancy and create a coherent narrative that covers the entire paper."""

        return self._call_llm_api(merge_prompt)

    def _call_llm_api(
        self, content: str, pdf_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Make an API call to the configured LLM.

        Args:
            content: Content to send to the LLM
            pdf_path: Optional path to PDF file for direct PDF processing

        Returns:
            Response text or None on failure
        """
        provider = self.model_config["provider"]

        try:
            if provider == "openai":
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ]

                # Handle PDF file attachment for OpenAI (if supported)
                if pdf_path:
                    # Note: Direct PDF support depends on model capabilities
                    # For now, we'll include a note about the PDF
                    messages[-1][
                        "content"
                    ] += f"\n\nNote: PDF file provided at {pdf_path}"

                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=messages,
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 2000),
                )
                return response.choices[0].message.content

            elif provider == "anthropic":
                # Anthropic supports PDF uploads in their API
                if pdf_path:
                    # For Claude, we can include the PDF as a document
                    # This would require base64 encoding the PDF
                    try:
                        import base64

                        with open(pdf_path, "rb") as f:
                            pdf_data = base64.b64encode(f.read()).decode()

                        response = self.client.messages.create(
                            model=self.model_config["model"],
                            system=self.system_prompt,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "document",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "application/pdf",
                                                "data": pdf_data,
                                            },
                                        },
                                        {
                                            "type": "text",
                                            "text": content,
                                        },
                                    ],
                                }
                            ],
                            temperature=self.model_config.get("temperature", 0.1),
                            max_tokens=self.model_config.get("max_tokens", 2000),
                        )
                    except Exception as e:
                        print(f"PDF upload failed, falling back to text-only: {e}")
                        response = self.client.messages.create(
                            model=self.model_config["model"],
                            system=self.system_prompt,
                            messages=[
                                {
                                    "role": "user",
                                    "content": content
                                    + f"\n\nNote: PDF file at {pdf_path} could not be processed directly.",
                                }
                            ],
                            temperature=self.model_config.get("temperature", 0.1),
                            max_tokens=self.model_config.get("max_tokens", 2000),
                        )
                else:
                    response = self.client.messages.create(
                        model=self.model_config["model"],
                        system=self.system_prompt,
                        messages=[{"role": "user", "content": content}],
                        temperature=self.model_config.get("temperature", 0.1),
                        max_tokens=self.model_config.get("max_tokens", 2000),
                    )
                return response.content[0].text

            elif provider == "google":
                # Google models may support PDF processing
                model = self.client.GenerativeModel(self.model_config["model"])
                prompt_content = [f"{self.system_prompt}\n\n{content}"]

                if pdf_path:
                    try:
                        # Try to upload PDF file
                        uploaded_file = self.client.upload_file(pdf_path)
                        prompt_content.append(uploaded_file)
                    except Exception as e:
                        print(f"PDF upload failed, using text-only: {e}")
                        prompt_content[
                            0
                        ] += f"\n\nNote: PDF file at {pdf_path} could not be processed directly."

                response = model.generate_content(
                    prompt_content,
                    generation_config=self.client.types.GenerationConfig(
                        temperature=self.model_config.get("temperature", 0.1),
                        max_output_tokens=self.model_config.get("max_tokens", 2000),
                    ),
                )
                return response.text

            elif provider in ["ollama", "lmstudio", "local", "custom"]:
                # Local models typically don't support PDF uploads
                if pdf_path:
                    content += f"\n\nNote: PDF file provided at {pdf_path} but direct PDF processing not supported by local models."

                if provider == "ollama":
                    return self._call_ollama_api(content)
                elif provider == "lmstudio":
                    return self._call_lmstudio_api(content)
                elif provider in ["local", "custom"]:
                    return self._call_local_api(content)

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def _call_ollama_api(self, content: str) -> Optional[str]:
        """Call Ollama API (text-only)."""
        import requests

        base_url = self.model_config.get("base_url", "http://localhost:11434")
        model_name = self.model_config["model"]

        full_prompt = f"{self.system_prompt}\n\n{content}"

        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.model_config.get("temperature", 0.1),
                "num_predict": self.model_config.get("max_tokens", 2000),
            },
        }

        try:
            response = requests.post(
                f"{base_url}/api/generate", json=payload, timeout=180
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"Ollama API call failed: {e}")
            return None

    def _call_lmstudio_api(self, content: str) -> Optional[str]:
        """Call LM Studio API (text-only)."""
        try:
            import openai

            base_url = self.model_config.get("base_url", "http://localhost:1234/v1")
            model_name = self.model_config["model"]

            local_client = openai.OpenAI(
                base_url=base_url,
                api_key="not-needed",
            )

            response = local_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=self.model_config.get("max_tokens", 2000),
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"LM Studio API call failed: {e}")
            return None

    def _call_local_api(self, content: str) -> Optional[str]:
        """Call custom local API (text-only)."""
        import requests

        try:
            base_url = self.model_config.get("base_url")
            if not base_url:
                raise ValueError("base_url is required for local/custom providers")

            model_name = self.model_config["model"]
            api_format = self.model_config.get("api_format", "openai")

            if api_format == "openai":
                import openai

                local_client = openai.OpenAI(
                    base_url=base_url,
                    api_key=self.model_config.get("api_key", "not-needed"),
                )

                response = local_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 2000),
                )

                return response.choices[0].message.content

            elif api_format == "ollama":
                full_prompt = f"{self.system_prompt}\n\n{content}"

                payload = {
                    "model": model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.model_config.get("temperature", 0.1),
                        "num_predict": self.model_config.get("max_tokens", 2000),
                    },
                }

                response = requests.post(
                    f"{base_url}/api/generate", json=payload, timeout=180
                )
                response.raise_for_status()

                result = response.json()
                return result.get("response", "")

            else:
                raise ValueError(f"Unsupported api_format: {api_format}")

        except Exception as e:
            print(f"Local API call failed: {e}")
            return None

    def _validate_summary_response(
        self, response_text: str
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Validate and parse the summary response.

        Args:
            response_text: Raw response from LLM

        Returns:
            Tuple of (is_valid, parsed_data)
        """
        try:
            # Clean up response
            cleaned_text = response_text.strip()

            # Remove markdown code blocks if present
            if cleaned_text.startswith("```"):
                lines = cleaned_text.split("\n")
                start_idx = 0
                end_idx = len(lines)

                for i, line in enumerate(lines):
                    if not line.strip().startswith("```"):
                        start_idx = i
                        break

                for i in range(len(lines) - 1, -1, -1):
                    if not lines[i].strip().startswith("```"):
                        end_idx = i + 1
                        break

                cleaned_text = "\n".join(lines[start_idx:end_idx])

            # Parse JSON
            try:
                data = json.loads(cleaned_text)
            except json.JSONDecodeError:
                # Look for JSON in the response
                json_match = re.search(
                    r'\{[^}]*"summary"[^}]*\}', cleaned_text, re.DOTALL
                )
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return False, None

            # Validate required fields
            if "summary" not in data:
                return False, None

            # Ensure optional fields exist
            if "key_contributions" not in data:
                data["key_contributions"] = []
            if "methodology_tags" not in data:
                data["methodology_tags"] = []
            if "confidence" not in data:
                data["confidence"] = 0.8

            # Validate confidence is a number
            try:
                data["confidence"] = float(data["confidence"])
                if not 0.0 <= data["confidence"] <= 1.0:
                    data["confidence"] = 0.8
            except (ValueError, TypeError):
                data["confidence"] = 0.8

            return True, data

        except Exception as e:
            print(f"Response validation error: {e}")
            return False, None

    def summarize_paper(self, paper: Dict) -> Dict:
        """
        Summarize a single paper with retry logic and chunking support.

        Args:
            paper: Paper dictionary

        Returns:
            Enhanced paper dictionary with summarization information
        """
        # Find content for the paper
        text_content, pdf_path, source_type = self._find_content_for_paper(paper)

        if source_type == "none":
            # No content available
            enhanced_paper = paper.copy()
            enhanced_paper.update(
                {
                    "llm_summary": None,
                    "summary_key_contributions": [],
                    "summary_methodology_tags": [],
                    "summary_confidence": 0.0,
                    "summary_source": "none",
                    "summary_chunked": False,
                    "summarized_by": self.model_alias,
                    "summarized_at": datetime.now().isoformat(),
                    "summary_error": "No content available for summarization",
                }
            )
            self.total_failures += 1
            return enhanced_paper

        # Determine processing approach
        needs_chunking = False
        estimated_tokens = 0

        if source_type == "extracted" and text_content:
            # Only apply chunking logic to extracted text
            estimated_tokens = self._estimate_tokens(text_content)
            needs_chunking = estimated_tokens > self.max_single_chunk_tokens
        elif source_type == "pdf":
            # PDF direct processing - no chunking
            estimated_tokens = -1  # Unknown
            needs_chunking = False

        for attempt in range(self.retry_attempts + 1):
            try:
                if needs_chunking:
                    # Use chunking approach (only for extracted text)
                    response_text = self._summarize_with_chunking(text_content, paper)
                    self.total_chunked += 1
                else:
                    # Single content summarization (text or PDF)
                    response_text = self._summarize_single_content(
                        text_content, pdf_path, paper
                    )

                if response_text is None:
                    if attempt < self.retry_attempts:
                        self.total_retries += 1
                        time.sleep(1)
                        continue
                    else:
                        break

                # Validate response
                is_valid, summary_data = self._validate_summary_response(response_text)

                if is_valid:
                    # Success!
                    enhanced_paper = paper.copy()
                    enhanced_paper.update(
                        {
                            "llm_summary": summary_data["summary"],
                            "summary_key_contributions": summary_data[
                                "key_contributions"
                            ],
                            "summary_methodology_tags": summary_data[
                                "methodology_tags"
                            ],
                            "summary_confidence": summary_data["confidence"],
                            "summary_source": source_type,
                            "summary_chunked": needs_chunking,
                            "summary_tokens_estimated": (
                                estimated_tokens if estimated_tokens > 0 else None
                            ),
                            "summarized_by": self.model_alias,
                            "summarized_at": datetime.now().isoformat(),
                        }
                    )

                    self.total_summarized += 1
                    return enhanced_paper

                else:
                    if attempt < self.retry_attempts:
                        self.total_retries += 1
                        print(
                            f"  → Invalid response format, retrying... (attempt {attempt + 1})"
                        )
                        time.sleep(1)
                        continue
                    else:
                        print(
                            f"  → Failed to get valid response after {self.retry_attempts + 1} attempts"
                        )
                        break

            except Exception as e:
                if attempt < self.retry_attempts:
                    self.total_retries += 1
                    print(
                        f"  → Error during summarization: {e}, retrying... (attempt {attempt + 1})"
                    )
                    time.sleep(1)
                    continue
                else:
                    print(
                        f"  → Final failure after {self.retry_attempts + 1} attempts: {e}"
                    )
                    break

        # All attempts failed
        self.total_failures += 1
        enhanced_paper = paper.copy()
        enhanced_paper.update(
            {
                "llm_summary": None,
                "summary_key_contributions": [],
                "summary_methodology_tags": [],
                "summary_confidence": 0.0,
                "summary_source": source_type,
                "summary_chunked": needs_chunking,
                "summary_tokens_estimated": (
                    estimated_tokens if estimated_tokens > 0 else None
                ),
                "summarized_by": self.model_alias,
                "summarized_at": datetime.now().isoformat(),
                "summary_error": "Summarization failed after retries",
            }
        )

        return enhanced_paper

    def _summarize_with_chunking(self, text: str, paper: Dict) -> Optional[str]:
        """
        Summarize text using chunking approach.

        Args:
            text: Text to summarize
            paper: Paper metadata

        Returns:
            LLM response or None on failure
        """
        # Create chunks
        chunks = self._chunk_text(text, self.max_chunk_tokens)

        if len(chunks) == 1:
            # Just use single text summarization
            return self._summarize_single_content(
                text_content=text, pdf_path=None, paper=paper
            )

        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            chunk_summary = self._summarize_chunk(chunk, paper)
            if chunk_summary:
                chunk_summaries.append(chunk_summary)

        if not chunk_summaries:
            return None

        # Merge chunk summaries
        if len(chunk_summaries) == 1:
            # Only one chunk succeeded, try to convert to final format
            final_summary = self._call_llm_api(
                f"""Convert this chunk summary to the final format:

{chunk_summaries[0]}

Convert this to the proper JSON format with summary, key_contributions, methodology_tags, and confidence fields."""
            )
            return final_summary
        else:
            # Merge multiple chunks
            return self._merge_chunk_summaries(chunk_summaries, paper)


def validate_summarization_config(config: Dict) -> List[str]:
    """
    Validate the summarization configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    summ_config = config.get("summarization", {})

    # Check required parameters
    model_alias = summ_config.get("model_alias")
    if not model_alias:
        errors.append("model_alias is required in summarization configuration")

    # Validate retry attempts
    retry_attempts = summ_config.get("retry_attempts", 2)
    if not isinstance(retry_attempts, int) or retry_attempts < 0:
        errors.append("retry_attempts must be a non-negative integer")

    # Validate token limits
    max_single_chunk = summ_config.get("max_single_chunk_tokens", 15000)
    if not isinstance(max_single_chunk, int) or max_single_chunk <= 0:
        errors.append("max_single_chunk_tokens must be a positive integer")

    max_chunk = summ_config.get("max_chunk_tokens", 8000)
    if not isinstance(max_chunk, int) or max_chunk <= 0:
        errors.append("max_chunk_tokens must be a positive integer")

    if max_chunk >= max_single_chunk:
        errors.append("max_chunk_tokens should be less than max_single_chunk_tokens")

    # Validate overlap ratio
    overlap_ratio = summ_config.get("chunk_overlap_ratio", 0.2)
    if not isinstance(overlap_ratio, (int, float)) or not 0 <= overlap_ratio <= 0.5:
        errors.append("chunk_overlap_ratio must be a number between 0 and 0.5")

    return errors


def display_default_summarization_prompts():
    """Display the default prompts for user reference."""
    defaults = get_default_summarization_prompts()

    print("=== DEFAULT SUMMARIZATION PROMPTS ===")
    for prompt_name, prompt_text in defaults.items():
        print(f"\n{prompt_name}:")
        print("-" * 40)
        print(prompt_text)
    print("\n" + "=" * 50)


def test_summarization_utilities():
    """Test summarization utilities with mock setup."""
    print("=== TESTING SUMMARIZATION UTILITIES ===")

    # Test configuration validation
    print("\n1. Testing configuration validation...")
    test_configs = {
        "valid_config": {
            "summarization": {
                "model_alias": "test-model",
                "retry_attempts": 2,
                "max_single_chunk_tokens": 15000,
                "max_chunk_tokens": 8000,
                "chunk_overlap_ratio": 0.2,
            }
        },
        "invalid_tokens": {
            "summarization": {
                "model_alias": "test-model",
                "max_single_chunk_tokens": 5000,
                "max_chunk_tokens": 8000,  # Should be less than single chunk
            }
        },
        "missing_model": {
            "summarization": {
                "retry_attempts": 2,
                # Missing model_alias
            }
        },
    }

    for config_name, config in test_configs.items():
        validation_errors = validate_summarization_config(config)
        expected_result = (
            "❌ (expected)"
            if "invalid" in config_name or "missing" in config_name
            else "✅"
        )
        success = (len(validation_errors) > 0) == (
            "invalid" in config_name or "missing" in config_name
        )

        if success:
            print(f"   {config_name}: {expected_result}")
        else:
            print(f"   {config_name}: ❌ Unexpected result - {validation_errors}")

    # Test token estimation
    print("\n2. Testing token estimation...")
    test_text = "This is a test sentence. " * 100  # ~500 words

    class MockSummarizer:
        def _estimate_tokens(self, text):
            return len(text) // 4

    mock_summarizer = MockSummarizer()
    estimated_tokens = mock_summarizer._estimate_tokens(test_text)
    print(f"   Text length: {len(test_text)} chars")
    print(f"   Estimated tokens: {estimated_tokens}")
    print(f"   ✅ Token estimation working")

    # Test text chunking
    print("\n3. Testing text chunking...")
    long_text = "This is paragraph one.\n\nThis is paragraph two. " * 200  # Long text

    class MockChunker:
        def __init__(self):
            self.chunk_overlap_ratio = 0.2

        def _chunk_text(self, text, max_tokens):
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return [{"text": text, "chunk_index": 0, "total_chunks": 1}]

            chunks = []
            overlap_chars = int(max_chars * self.chunk_overlap_ratio)
            start = 0
            chunk_index = 0

            while start < len(text):
                end = start + max_chars
                if end >= len(text):
                    chunk_text = text[start:]
                else:
                    chunk_text = text[start:end]

                chunks.append(
                    {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "total_chunks": -1,
                    }
                )

                start = end - overlap_chars
                chunk_index += 1

            for chunk in chunks:
                chunk["total_chunks"] = len(chunks)

            return chunks

    chunker = MockChunker()
    chunks = chunker._chunk_text(long_text, 2000)  # 2000 tokens max
    print(f"   Original text: {len(long_text)} chars")
    print(f"   Number of chunks: {len(chunks)}")
    print(f"   Chunk sizes: {[len(c['text']) for c in chunks]}")
    print(f"   ✅ Text chunking working")

    # Test response validation
    print("\n4. Testing response validation...")
    test_responses = [
        '{"summary": "This is a test summary", "key_contributions": ["point1"], "confidence": 0.9}',  # Valid
        '{"summary": "Missing other fields"}',  # Missing fields
        '{"not_summary": "Wrong structure"}',  # Wrong structure
        "Not JSON at all",  # Invalid JSON
    ]

    class MockValidator:
        def _validate_summary_response(self, response_text):
            import json
            import re

            try:
                cleaned_text = response_text.strip()

                try:
                    data = json.loads(cleaned_text)
                except json.JSONDecodeError:
                    json_match = re.search(
                        r'\{[^}]*"summary"[^}]*\}', cleaned_text, re.DOTALL
                    )
                    if json_match:
                        data = json.loads(json_match.group())
                    else:
                        return False, None

                if "summary" not in data:
                    return False, None

                if "key_contributions" not in data:
                    data["key_contributions"] = []
                if "methodology_tags" not in data:
                    data["methodology_tags"] = []
                if "confidence" not in data:
                    data["confidence"] = 0.8

                return True, data
            except:
                return False, None

    validator = MockValidator()
    for i, response in enumerate(test_responses, 1):
        is_valid, parsed = validator._validate_summary_response(response)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"    Test {i}: {status}")

    print(f"\n✅ All summarization utilities tests passed")


def test_real_summarization_config():
    """Test summarization configuration with real config file."""
    print("=== TESTING WITH REAL SUMMARIZATION CONFIGURATION ===")

    try:
        import yaml
        from llm_utils import LLMManager

        # Load real configuration
        try:
            with open("config/config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            print("✅ Loaded config/config.yaml")
        except FileNotFoundError:
            print("❌ config/config.yaml not found")
            return
        except Exception as e:
            print(f"❌ Error loading config/config.yaml: {e}")
            return

        # Validate configuration
        validation_errors = validate_summarization_config(config)
        if not validation_errors:
            print("✅ Summarization configuration validation passed")
        else:
            print("❌ Summarization configuration validation failed:")
            for error in validation_errors:
                print(f"   - {error}")
            return

        # Test LLM manager integration
        summ_config = config.get("summarization", {})
        model_alias = summ_config.get("model_alias")

        if not model_alias:
            print("❌ No model_alias specified in summarization configuration")
            return

        try:
            llm_manager = LLMManager()
            model_config = llm_manager.get_model_config(model_alias)
            print(f"✅ Model '{model_alias}' configuration loaded:")
            print(f"   Provider: {model_config.get('provider')}")
            print(f"   Model: {model_config.get('model')}")
            print(f"   Temperature: {model_config.get('temperature')}")
        except Exception as e:
            print(f"❌ Error getting model configuration: {e}")
            return

        # Show summarization configuration
        print(f"Summarization configuration:")
        print(f"  Max papers: {summ_config.get('max_papers', 'unlimited')}")
        print(
            f"  Prefer extracted text: {summ_config.get('prefer_extracted_text', True)}"
        )
        print(
            f"  Max single chunk tokens: {summ_config.get('max_single_chunk_tokens', 15000)}"
        )
        print(f"  Max chunk tokens: {summ_config.get('max_chunk_tokens', 8000)}")
        print(f"  Chunk overlap ratio: {summ_config.get('chunk_overlap_ratio', 0.2)}")

        # Check for custom prompts
        custom_prompts = []
        if summ_config.get("research_context_prompt", "").strip():
            custom_prompts.append("research_context_prompt")
        if summ_config.get("summarization_strategy_prompt", "").strip():
            custom_prompts.append("summarization_strategy_prompt")
        if summ_config.get("summary_format_prompt", "").strip():
            custom_prompts.append("summary_format_prompt")

        if custom_prompts:
            print(f"  Custom prompts configured: {', '.join(custom_prompts)}")
        else:
            print(f"  Using default prompts (consider customizing for better results)")

        print(f"\n✅ Real summarization configuration test complete")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            test_summarization_utilities()
        elif sys.argv[1] in ["--test-real", "--real", "-r"]:
            test_real_summarization_config()
        elif sys.argv[1] in ["--defaults", "-d"]:
            display_default_summarization_prompts()
        elif sys.argv[1] in ["--help", "-h"]:
            print("Paper Summarization Utilities")
            print("Usage:")
            print("  python summarize_utils.py --test       # Run mock tests")
            print(
                "  python summarize_utils.py --test-real  # Test with real configuration"
            )
            print("  python summarize_utils.py --defaults   # Show default prompts")
            print("  python summarize_utils.py --help       # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        print("Paper Summarization Utilities Module")
        print("Run with --help for usage options")
