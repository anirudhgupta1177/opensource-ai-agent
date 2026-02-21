"""
ReAct (Reasoning + Acting) agent for multi-step research.
Implements think-act-observe loop for complex queries.
"""
from __future__ import annotations


import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from src.config import get_config
from src.context.budget import build_context_with_sources
from src.crawl.fetcher import fetch_and_extract
from src.llm.client import chat
from src.llm.prompts import REACT_SYSTEM_PROMPT, get_react_conclude_prompt
from src.search.duckduckgo_search import search as ddg_search

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Available agent actions."""
    SEARCH = "search"
    FETCH = "fetch"
    ANALYZE = "analyze"
    VERIFY = "verify"
    CONCLUDE = "conclude"


@dataclass
class AgentStep:
    """Record of a single agent step."""
    step_number: int
    thought: str
    action: ActionType
    action_input: str
    observation: str
    success: bool = True


@dataclass
class AgentContext:
    """Running context for the agent."""
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    gathered_info: list[str] = field(default_factory=list)
    search_results: list[dict[str, Any]] = field(default_factory=list)


class ReActAgent:
    """
    ReAct agent implementing think-act-observe loop.

    The agent reasons about what to do, takes an action using available tools,
    observes the result, and repeats until it can provide a final answer.
    """

    def __init__(self, max_steps: Optional[int] = None):
        config = get_config()
        self.max_steps = max_steps or config.agent.max_react_steps

        # Register available tools
        self.tools: dict[ActionType, Callable] = {
            ActionType.SEARCH: self._tool_search,
            ActionType.FETCH: self._tool_fetch,
            ActionType.ANALYZE: self._tool_analyze,
            ActionType.VERIFY: self._tool_verify,
            ActionType.CONCLUDE: self._tool_conclude,
        }

    def run(self, query: str) -> dict[str, Any]:
        """
        Run the ReAct agent on a query.

        Returns:
            Dict with content, sources, steps, and metadata
        """
        context = AgentContext(query=query)
        logger.info("Starting ReAct agent for query: %s", query[:100])

        for step_num in range(1, self.max_steps + 1):
            try:
                # Get next action from LLM
                thought, action, action_input = self._think(context)

                logger.debug(
                    "Step %d: Action=%s, Input=%s",
                    step_num,
                    action.value,
                    action_input[:50] if action_input else "None"
                )

                # Execute action
                observation, success = self._act(action, action_input, context)

                # Record step
                step = AgentStep(
                    step_number=step_num,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation[:1000],  # Truncate long observations
                    success=success,
                )
                context.steps.append(step)

                # Check if done
                if action == ActionType.CONCLUDE:
                    logger.info("Agent concluded after %d steps", step_num)
                    return self._build_result(context, observation)

            except Exception as e:
                logger.error("ReAct step %d failed: %s", step_num, str(e)[:200])
                # Record failed step and continue
                context.steps.append(AgentStep(
                    step_number=step_num,
                    thought="Error occurred",
                    action=ActionType.ANALYZE,
                    action_input="",
                    observation=f"Error: {str(e)[:500]}",
                    success=False,
                ))

        # Max steps reached, force conclusion
        logger.warning("Agent reached max steps (%d), forcing conclusion", self.max_steps)
        return self._force_conclude(context)

    def _think(self, context: AgentContext) -> tuple[str, ActionType, str]:
        """
        Generate next thought and action using LLM.

        Returns:
            Tuple of (thought, action_type, action_input)
        """
        # Build prompt with history
        history = self._format_history(context)

        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {context.query}\n\n{history}\n\nWhat's your next step?"},
        ]

        response = chat(messages, max_tokens=512)
        return self._parse_thought_action(response)

    def _parse_thought_action(self, response: str) -> tuple[str, ActionType, str]:
        """Parse LLM response to extract thought, action, and input."""
        # Default values
        thought = ""
        action = ActionType.CONCLUDE
        action_input = ""

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()

            if line.lower().startswith("thought:"):
                thought = line[8:].strip()
            elif line.lower().startswith("action:"):
                action_str = line[7:].strip().lower()
                try:
                    action = ActionType(action_str)
                except ValueError:
                    # Try fuzzy matching
                    for at in ActionType:
                        if at.value in action_str:
                            action = at
                            break
            elif line.lower().startswith("action input:"):
                action_input = line[13:].strip()

        # If no thought parsed, use the whole response as thought
        if not thought and response:
            thought = response[:200]

        return thought, action, action_input

    def _act(
        self,
        action: ActionType,
        action_input: str,
        context: AgentContext
    ) -> tuple[str, bool]:
        """
        Execute an action and return observation.

        Returns:
            Tuple of (observation_string, success_bool)
        """
        tool = self.tools.get(action)
        if not tool:
            return f"Unknown action: {action}", False

        try:
            observation = tool(action_input, context)
            return observation, True
        except Exception as e:
            return f"Action failed: {str(e)[:200]}", False

    def _tool_search(self, query: str, context: AgentContext) -> str:
        """Search the web for information."""
        if not query:
            return "Error: No search query provided"

        results = ddg_search(query, max_results=8)

        if not results:
            return "No search results found"

        # Add to context
        context.search_results.extend(results)

        # Format results for observation
        formatted = []
        for i, r in enumerate(results[:5], 1):
            formatted.append(f"{i}. {r.get('title', 'No title')}\n   URL: {r.get('url', '')}\n   {r.get('snippet', '')[:150]}")

        return f"Found {len(results)} results:\n\n" + "\n\n".join(formatted)

    def _tool_fetch(self, url: str, context: AgentContext) -> str:
        """Fetch and extract content from a URL."""
        if not url:
            return "Error: No URL provided"

        # Clean URL
        url = url.strip().strip('"\'')

        content = fetch_and_extract(url)

        if content == "Content unavailable":
            return f"Could not extract content from {url}"

        # Add to gathered info and sources
        context.gathered_info.append(f"[From {url}]:\n{content[:2000]}")
        context.sources.append({
            "url": url,
            "title": "",
            "snippet": content[:200],
            "text": content,
        })

        return f"Extracted {len(content)} characters from {url}:\n\n{content[:1500]}..."

    def _tool_analyze(self, text: str, context: AgentContext) -> str:
        """Analyze gathered information."""
        if not context.gathered_info:
            return "No information gathered yet to analyze. Use 'search' and 'fetch' first."

        # Build analysis prompt
        info_summary = "\n\n---\n\n".join(context.gathered_info[-5:])  # Last 5 pieces

        messages = [
            {"role": "system", "content": "Analyze the provided information and summarize key findings relevant to the query."},
            {"role": "user", "content": f"Query: {context.query}\n\nInformation:\n{info_summary}\n\nProvide a brief analysis:"},
        ]

        analysis = chat(messages, max_tokens=512)
        context.gathered_info.append(f"[Analysis]:\n{analysis}")

        return analysis

    def _tool_verify(self, claim: str, context: AgentContext) -> str:
        """Verify a claim against gathered sources."""
        if not claim:
            return "Error: No claim to verify"

        if not context.sources:
            return "No sources available to verify against. Fetch some URLs first."

        # Check claim against sources
        supporting = []
        for source in context.sources:
            text = source.get("text", "").lower()
            claim_words = claim.lower().split()
            matches = sum(1 for word in claim_words if word in text)
            match_ratio = matches / len(claim_words) if claim_words else 0

            if match_ratio > 0.5:
                supporting.append(source.get("url", "unknown"))

        if supporting:
            return f"Claim appears supported by {len(supporting)} source(s): {', '.join(supporting[:3])}"
        else:
            return "Claim could not be verified in the gathered sources"

    def _tool_conclude(self, answer: str, context: AgentContext) -> str:
        """Provide final answer with citations."""
        if not answer and not context.gathered_info:
            return "Cannot conclude without gathered information"

        # If answer provided, use it directly
        if answer and len(answer) > 50:
            return answer

        # Otherwise, generate conclusion from gathered info
        prompt = get_react_conclude_prompt(
            query=context.query,
            context="\n\n".join(context.gathered_info[-10:])
        )

        messages = [
            {"role": "system", "content": "You are a research assistant providing final answers with citations."},
            {"role": "user", "content": prompt},
        ]

        conclusion = chat(messages, max_tokens=1024)
        return conclusion

    def _format_history(self, context: AgentContext) -> str:
        """Format step history for LLM context."""
        if not context.steps:
            return "No previous steps."

        parts = []
        for step in context.steps[-5:]:  # Last 5 steps
            parts.append(
                f"Step {step.step_number}:\n"
                f"Thought: {step.thought}\n"
                f"Action: {step.action.value}\n"
                f"Action Input: {step.action_input}\n"
                f"Observation: {step.observation[:500]}"
            )

        return "\n\n".join(parts)

    def _build_result(self, context: AgentContext, final_content: str) -> dict[str, Any]:
        """Build final result dict."""
        return {
            "content": final_content,
            "sources": context.sources,
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action.value,
                    "success": s.success,
                }
                for s in context.steps
            ],
            "total_steps": len(context.steps),
            "error": None,
        }

    def _force_conclude(self, context: AgentContext) -> dict[str, Any]:
        """Force a conclusion when max steps reached."""
        # Generate best-effort conclusion
        if context.gathered_info:
            prompt = get_react_conclude_prompt(
                query=context.query,
                context="\n\n".join(context.gathered_info[-10:])
            )

            try:
                messages = [
                    {"role": "system", "content": "Provide a final answer based on gathered information. Note that the research was incomplete."},
                    {"role": "user", "content": prompt},
                ]
                content = chat(messages, max_tokens=1024)
                content = f"*Note: Research reached step limit. Results may be incomplete.*\n\n{content}"
            except Exception:
                content = "Research incomplete: reached maximum steps without conclusion."
        else:
            content = "Unable to find relevant information for this query."

        return {
            "content": content,
            "sources": context.sources,
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action.value,
                    "success": s.success,
                }
                for s in context.steps
            ],
            "total_steps": len(context.steps),
            "error": "Reached maximum steps",
        }


def run_react_agent(query: str, max_steps: Optional[int] = None) -> dict[str, Any]:
    """
    Convenience function to run ReAct agent.

    Args:
        query: Research query
        max_steps: Maximum reasoning steps

    Returns:
        Result dict with content, sources, steps
    """
    agent = ReActAgent(max_steps=max_steps)
    return agent.run(query)
