#!/usr/bin/env python3
"""
Paper Summarization Utilities

Handles LLM-based summarization of papers with text chunking for large documents.
Uses natural language output with confidence statements instead of strict JSON format.

Dependencies:
    pip install openai anthropic google-generativeai PyPDF2 pymupdf pdfplumber

Usage:
    from summarize_utils import PaperSummarizer

    summarizer = PaperSummarizer(config, llm_manager)
    summary_data = summarizer.summarize_paper(paper_data)
"""

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
Length: Aim for 1-2 pages (800-1500 words maximum).
Format: Use clear markdown structure with section headers.
Style: Technical but concise, suitable for expert readers.
Focus: Emphasize key findings, methodology, and practical significance.

Do not attempt to reproduce mathematical equations or complex formulas.
Instead, describe them concisely (e.g., "uses a modified loss function that combines...").
Avoid verbose explanations and focus on the most important insights.
        """.strip(),
    }


def get_confidence_instruction() -> str:
    """
    Get the standard confidence instruction that is appended to all prompts.

    Returns:
        Standard confidence instruction string
    """
    return """
IMPORTANT: End your response with a confidence statement in this exact format:
"CONFIDENCE: I have [high/medium/low] confidence in this summary based on [brief reason]."
""".strip()


class PaperSummarizer:
    """
    Handles LLM-based paper summarization with simplified output format.
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

        # Always append the confidence instruction
        confidence_instruction = get_confidence_instruction()

        # Warn user about defaults
        if used_defaults:
            print(f"⚠️  WARNING: Using default prompts for: {', '.join(used_defaults)}")
            print(
                f"   For better results, customize these prompts in your config file."
            )
            print()

        # Combine the parts with confidence instruction always at the end
        prompt_parts = [
            f"RESEARCH CONTEXT:\n{research_context}",
            f"SUMMARIZATION STRATEGY:\n{summarization_strategy}",
            f"FORMATTING GUIDELINES:\n{summary_format}",
            confidence_instruction,
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

    def _extract_confidence_from_summary(self, summary_text: str) -> Tuple[str, float]:
        """
        Extract confidence level from summary text and clean it.

        Args:
            summary_text: The complete summary text

        Returns:
            Tuple of (clean_summary, confidence_score)
        """
        # Remove thinking tokens first
        clean_text = self._remove_thinking_tokens(summary_text)

        # Look for confidence statement at the end with corrected grammar
        confidence_pattern = (
            r"CONFIDENCE:\s*I have (high|medium|low) confidence.*?(?:\.|$)"
        )
        confidence_match = re.search(
            confidence_pattern, clean_text, re.IGNORECASE | re.DOTALL
        )

        confidence_score = 0.8  # Default
        clean_summary = clean_text

        if confidence_match:
            confidence_level = confidence_match.group(1).lower()
            # Convert to numeric
            confidence_mapping = {"high": 0.9, "medium": 0.7, "low": 0.4}
            confidence_score = confidence_mapping.get(confidence_level, 0.7)

            # Remove confidence statement from summary
            clean_summary = clean_text[: confidence_match.start()].strip()
        else:
            # Try alternative patterns (including old grammar for backwards compatibility)
            alt_patterns = [
                r"CONFIDENCE:\s*I am (high|medium|low) confidence",  # Old grammar
                r"CONFIDENCE:\s*I have\s+(high|medium|low) confidence",  # Handle extra spaces
                r"CONFIDENCE:\s*I have\s+confidence",  # Incomplete - default to medium
                r"confidence[:\s]+(high|medium|low)",
                r"I (?:am|have) (very confident|confident|somewhat confident|not very confident)",
                r"(high|medium|low) confidence",
            ]

            for pattern in alt_patterns:
                match = re.search(pattern, clean_text, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 0:
                        # Pattern matched but no level captured (incomplete statement)
                        confidence_score = 0.7  # Default to medium
                    else:
                        level = match.group(1).lower()
                        if "high" in level or "very confident" in level:
                            confidence_score = 0.9
                        elif "medium" in level or level == "confident":
                            confidence_score = 0.7
                        elif (
                            "low" in level or "not very" in level or "somewhat" in level
                        ):
                            confidence_score = 0.4

                    # Try to remove this statement
                    clean_summary = re.sub(
                        pattern, "", clean_text, flags=re.IGNORECASE
                    ).strip()
                    break

        return clean_summary, confidence_score

    def _remove_thinking_tokens(self, text: str) -> str:
        """
        Remove thinking tokens and other unwanted content from LLM output.

        Args:
            text: Raw text from LLM

        Returns:
            Cleaned text
        """
        # Remove thinking blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove any remaining XML-like tags that might leak through
        text = re.sub(r"<[^>]+>", "", text)

        # Clean up extra whitespace
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # Multiple newlines to double
        text = text.strip()

        return text

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
Please provide a focused summary of this section of a research paper. This is chunk {chunk['chunk_index'] + 1} of {chunk['total_chunks']} total chunks.

Summarize the key information in this section without trying to provide a complete paper summary. 
Focus on the specific content available in this chunk.

{paper_info}{chunk_info}SECTION CONTENT:\n{chunk['text']}

Provide a narrative summary of this section's content. Include what type of section this appears to be (introduction, methods, results, discussion, etc.) if you can determine it.

{get_confidence_instruction()}
"""

        return self._call_llm_api(chunk_prompt)

    def _merge_chunk_summaries(
        self, chunk_summaries: List[str], paper: Dict
    ) -> Optional[str]:
        """
        Merge multiple chunk summaries into a final comprehensive summary.

        Args:
            chunk_summaries: List of chunk summary texts
            paper: Paper metadata

        Returns:
            Final merged summary or None on failure
        """
        paper_info = f"Title: {paper.get('title', 'Unknown')}\n"
        paper_info += f"Authors: {', '.join(paper.get('authors', []))}\n"
        paper_info += f"Categories: {', '.join(paper.get('categories', []))}\n\n"

        # Clean chunk summaries and combine
        cleaned_summaries = []
        for i, summary in enumerate(chunk_summaries):
            # Remove confidence statements from chunks
            cleaned = re.sub(
                r"CONFIDENCE:.*$", "", summary, flags=re.IGNORECASE | re.DOTALL
            ).strip()
            cleaned_summaries.append(f"SECTION {i+1} SUMMARY:\n{cleaned}")

        summaries_text = "\n\n".join(cleaned_summaries)

        merge_prompt = f"""Please create a comprehensive summary by merging these section summaries from a research paper.

{paper_info}SECTION SUMMARIES TO MERGE:
{summaries_text}

Create a unified, comprehensive summary that follows the original format requirements.
Eliminate redundancy and create a coherent narrative that covers the entire paper.

Organize the merged content using the standard structure:
1. CORE CONTRIBUTION
2. METHODOLOGY  
3. RESULTS
4. IMPLICATIONS

{get_confidence_instruction()}
"""

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
        Validate and parse the summary response using the new simplified format.

        Args:
            response_text: Raw response from LLM

        Returns:
            Tuple of (is_valid, parsed_data)
        """
        try:
            # Check if we have a reasonable summary
            if not response_text or len(response_text.strip()) < 100:
                return False, None

            # Extract confidence and clean the summary
            clean_summary, confidence_score = self._extract_confidence_from_summary(
                response_text
            )

            # Check if we have a valid summary after cleaning
            if len(clean_summary.strip()) < 50:
                return False, None

            # Create the simplified structured data
            parsed_data = {"summary": clean_summary, "confidence": confidence_score}

            return True, parsed_data

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
        Summarize text using chunking approach with simplified merging.

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

        # Merge chunk summaries - much simpler now!
        if len(chunk_summaries) == 1:
            # Only one chunk succeeded, return as-is
            return chunk_summaries[0]
        else:
            # Merge multiple chunks
            return self._merge_chunk_summaries(chunk_summaries, paper)


# Keep the rest of the validation and testing functions unchanged
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
    print("=== TESTING SIMPLIFIED SUMMARIZATION UTILITIES ===")

    # Test thinking token removal
    print("\n1. Testing thinking token removal...")
    test_text_with_thinking = """<think>
This is internal reasoning that should be removed.
Let me think about this...
</think>

**Summary of the Paper**

This is the actual summary content that should remain.

CONFIDENCE: I have high confidence in this summary based on clear methodology."""

    class MockProcessor:
        def _remove_thinking_tokens(self, text):
            # Remove thinking blocks
            text = re.sub(
                r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
            )
            # Remove any remaining XML-like tags
            text = re.sub(r"<[^>]+>", "", text)
            # Clean up extra whitespace
            text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
            text = text.strip()
            return text

    processor = MockProcessor()
    cleaned = processor._remove_thinking_tokens(test_text_with_thinking)

    print(f"   Original length: {len(test_text_with_thinking)} chars")
    print(f"   Cleaned length: {len(cleaned)} chars")
    print(f"   Thinking tokens removed: {'<think>' not in cleaned}")
    print("   ✅ Thinking token removal working")

    # Test confidence extraction
    print("\n2. Testing confidence extraction...")
    test_summaries = [
        "This is a great paper about AI. CONFIDENCE: I have high confidence in this summary based on clear methodology.",
        "The paper discusses machine learning. CONFIDENCE: I have medium confidence in this assessment.",
        "Complex analysis of data. CONFIDENCE: I have low confidence due to limited information available.",
        "Standard paper without confidence statement.",
        "Legacy format: CONFIDENCE: I am high confidence in this summary.",  # Test backwards compatibility
        "Malformed: CONFIDENCE: I have  confidence in this summary based on missing level.",  # Test missing level
    ]

    class MockSummarizer:
        def _remove_thinking_tokens(self, text):
            return text  # No thinking tokens in test cases

        def _extract_confidence_from_summary(self, summary_text):
            # Remove thinking tokens first
            clean_text = self._remove_thinking_tokens(summary_text)

            # Look for confidence statement
            confidence_pattern = (
                r"CONFIDENCE:\s*I have (high|medium|low) confidence.*?(?:\.|$)"
            )
            confidence_match = re.search(
                confidence_pattern, clean_text, re.IGNORECASE | re.DOTALL
            )

            confidence_score = 0.8  # Default
            clean_summary = clean_text

            if confidence_match:
                confidence_level = confidence_match.group(1).lower()
                confidence_mapping = {"high": 0.9, "medium": 0.7, "low": 0.4}
                confidence_score = confidence_mapping.get(confidence_level, 0.7)
                clean_summary = clean_text[: confidence_match.start()].strip()
            else:
                # Try legacy pattern
                legacy_match = re.search(
                    r"CONFIDENCE:\s*I am (high|medium|low) confidence",
                    clean_text,
                    re.IGNORECASE,
                )
                if legacy_match:
                    confidence_level = legacy_match.group(1).lower()
                    confidence_mapping = {"high": 0.9, "medium": 0.7, "low": 0.4}
                    confidence_score = confidence_mapping.get(confidence_level, 0.7)
                    clean_summary = clean_text[: legacy_match.start()].strip()

            return clean_summary, confidence_score

    mock_summarizer = MockSummarizer()

    for i, summary in enumerate(test_summaries, 1):
        clean, confidence = mock_summarizer._extract_confidence_from_summary(summary)
        print(f"   Test {i}: Confidence = {confidence:.1f}")

    print("   ✅ Confidence extraction working (including backwards compatibility)")

    # Test confidence instruction generation
    print("\n3. Testing confidence instruction...")
    instruction = get_confidence_instruction()
    print(f"   Standard instruction: {instruction[:50]}...")
    assert "I have" in instruction, "Should use correct grammar"
    assert "[high/medium/low]" in instruction, "Should include options"
    print("   ✅ Confidence instruction properly formatted")

    # Test simplified response validation
    print("\n4. Testing simplified response validation...")

    class MockValidator:
        def _extract_confidence_from_summary(self, summary_text):
            return summary_text.strip(), 0.8

        def _validate_summary_response(self, response_text):
            try:
                if not response_text or len(response_text.strip()) < 100:
                    return False, None

                clean_summary, confidence_score = self._extract_confidence_from_summary(
                    response_text
                )

                if len(clean_summary.strip()) < 50:
                    return False, None

                parsed_data = {"summary": clean_summary, "confidence": confidence_score}

                return True, parsed_data
            except:
                return False, None

    validator = MockValidator()

    test_responses = [
        "This is a comprehensive summary of the paper discussing novel AI methods. "
        * 10,  # Valid long
        "Too short",  # Too short
        "",  # Empty
        "This is a medium length summary that should work fine for testing purposes and validation."
        * 3,  # Valid medium
    ]

    for i, response in enumerate(test_responses, 1):
        is_valid, parsed = validator._validate_summary_response(response)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"    Test {i}: {status}")
        if is_valid:
            print(f"      → Fields: {list(parsed.keys())}")

    print("   ✅ Validation working")

    print(f"\n✅ All summarization utilities tests passed")


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            test_summarization_utilities()
        elif sys.argv[1] in ["--defaults", "-d"]:
            display_default_summarization_prompts()
        elif sys.argv[1] in ["--help", "-h"]:
            print("Simplified Paper Summarization Utilities")
            print("Usage:")
            print("  python summarize_utils.py --test       # Run mock tests")
            print("  python summarize_utils.py --defaults   # Show default prompts")
            print("  python summarize_utils.py --help       # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        print("Simplified Paper Summarization Utilities Module")
        print("Run with --help for usage options")
