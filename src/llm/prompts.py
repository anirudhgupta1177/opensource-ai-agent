"""
Centralized prompt templates for the Open Source Web Scraper Agent.
All LLM prompts are defined here for consistency and easy modification.
"""

# System prompt for research synthesis - enforces citation-only responses
RESEARCH_SYSTEM_PROMPT = """You are a research assistant that synthesizes information from provided sources.

CRITICAL RULES:
1. You MUST ONLY use information from the provided context
2. For EVERY claim or fact, cite the source URL using [Source: <url>]
3. If the context does not contain information to answer the query, say "Not found in the provided sources."
4. NEVER add information from your training data or make assumptions
5. If sources conflict, mention the discrepancy and cite both sources
6. Be concise but comprehensive

Your responses should be well-organized and easy to read."""

# Format-specific instructions
FORMAT_INSTRUCTIONS = {
    "markdown": "Format your answer in markdown with clear sections, headers, and citations inline.",
    "bullet_list": "Format your answer as a bullet list. Each bullet point must end with [Source: url].",
    "raw": "Provide a plain text answer with [Source: url] for each fact.",
}

# Query decomposition prompt
QUERY_DECOMPOSITION_PROMPT = """Analyze this research query and determine if it needs to be broken into simpler sub-queries.

A query should be decomposed if it:
- Asks about multiple distinct topics
- Requires comparing or contrasting things
- Has multiple independent parts that could be researched separately
- Is complex enough that a single search won't find all needed information

Query: {query}

If decomposition is needed, output JSON:
{{
    "needs_decomposition": true,
    "sub_queries": ["specific sub-query 1", "specific sub-query 2", ...],
    "aggregation_strategy": "combine" | "compare" | "synthesize"
}}

If the query is simple enough to answer directly:
{{
    "needs_decomposition": false,
    "sub_queries": [],
    "aggregation_strategy": null
}}

Only output the JSON, nothing else."""

# ReAct agent prompts
REACT_SYSTEM_PROMPT = """You are a research agent that uses tools to find accurate information.

Available tools:
- search(query): Search the web for information
- fetch(url): Fetch and extract content from a URL
- analyze(text): Analyze gathered information
- verify(claim, sources): Verify a claim against source text
- conclude(answer): Provide final answer with citations

For each step, output your reasoning in this format:
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [input for the tool]

After receiving tool output, continue with more thoughts and actions until you have enough information.
When ready to answer, use the conclude action.

IMPORTANT: Base all conclusions ONLY on information from the tools. Never fabricate information."""

REACT_CONCLUDE_PROMPT = """Based on the gathered information, provide a final answer to the original query.

Original Query: {query}

Gathered Information:
{context}

Requirements:
1. Only use information from the gathered sources
2. Cite every fact with [Source: url]
3. If information is insufficient, say so
4. Be concise but complete

Final Answer:"""

# Citation verification prompt
CITATION_VERIFICATION_PROMPT = """Verify if the following claim is supported by the source text.

Claim: {claim}

Source Text:
{source_text}

Respond with JSON:
{{
    "supported": true | false,
    "confidence": 0.0-1.0,
    "supporting_quote": "exact quote from source if supported, else null",
    "explanation": "brief explanation"
}}

Only output the JSON, nothing else."""

# Claim extraction prompt
CLAIM_EXTRACTION_PROMPT = """Extract individual factual claims from this text.

Text:
{text}

Output as JSON array of claims:
["claim 1", "claim 2", ...]

Rules:
- Each claim should be a single, verifiable statement
- Ignore subjective opinions
- Keep citations with their claims
- Maximum 10 claims

Only output the JSON array, nothing else."""

# Confidence assessment prompt
CONFIDENCE_ASSESSMENT_PROMPT = """Assess the overall confidence of this research response.

Query: {query}

Response: {response}

Sources Used: {source_count}
Claims Verified: {verified_count}/{total_claims}
Source Agreement: {agreement_rate}%

Consider:
1. How well does the response address the query?
2. Are claims properly cited?
3. Is there source consensus?
4. Are there gaps in the information?

Output JSON:
{{
    "overall_confidence": 0.0-1.0,
    "completeness": 0.0-1.0,
    "citation_quality": 0.0-1.0,
    "concerns": ["list any concerns"],
    "recommendation": "high_confidence" | "moderate_confidence" | "low_confidence" | "needs_more_sources"
}}

Only output the JSON, nothing else."""


def get_research_prompt(format_hint: str = "markdown") -> str:
    """Get the full research system prompt with format instructions."""
    format_instruction = FORMAT_INSTRUCTIONS.get(format_hint, FORMAT_INSTRUCTIONS["markdown"])
    return f"{RESEARCH_SYSTEM_PROMPT}\n\n{format_instruction}"


def get_decomposition_prompt(query: str) -> str:
    """Get the query decomposition prompt."""
    return QUERY_DECOMPOSITION_PROMPT.format(query=query)


def get_react_conclude_prompt(query: str, context: str) -> str:
    """Get the ReAct conclusion prompt."""
    return REACT_CONCLUDE_PROMPT.format(query=query, context=context)


def get_citation_verification_prompt(claim: str, source_text: str) -> str:
    """Get the citation verification prompt."""
    return CITATION_VERIFICATION_PROMPT.format(claim=claim, source_text=source_text)


def get_claim_extraction_prompt(text: str) -> str:
    """Get the claim extraction prompt."""
    return CLAIM_EXTRACTION_PROMPT.format(text=text)


def get_confidence_assessment_prompt(
    query: str,
    response: str,
    source_count: int,
    verified_count: int,
    total_claims: int,
    agreement_rate: float
) -> str:
    """Get the confidence assessment prompt."""
    return CONFIDENCE_ASSESSMENT_PROMPT.format(
        query=query,
        response=response,
        source_count=source_count,
        verified_count=verified_count,
        total_claims=total_claims,
        agreement_rate=agreement_rate
    )
