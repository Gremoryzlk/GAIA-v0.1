"""Prompt templates for LLM task processing in GAIA v7.3."""

from typing import Dict, Any, Optional


# Template for parsing natural language tasks into structured format
TASK_PARSE_TEMPLATE = """You are a task parsing assistant. Extract structured information from the user's request.

User request: {user_input}

Extract the following fields in JSON format:
- task_type: The category of task (e.g., "query", "analysis", "generation", "action")
- priority: Integer priority level (0-10, where 10 is highest)
- entities: List of key entities mentioned (people, places, concepts)
- intent: The primary action or goal the user wants to accomplish
- constraints: Any limitations or requirements mentioned
- expected_output: What form the result should take

Respond with valid JSON only, no additional text.
"""

# Template for fallback when primary task processing fails
FALLBACK_TEMPLATE = """The primary task processing failed. Please provide a safe, helpful response.

Original request: {user_input}
Error context: {error_context}

Provide a brief, helpful response that:
1. Acknowledges the issue
2. Offers alternative approaches if possible
3. Maintains safety and ethical guidelines

Keep your response concise and actionable.
"""


def format_task_prompt(
    user_input: str,
    template_type: str = "task_parse",
    error_context: Optional[str] = None,
    **kwargs: Any
) -> str:
    """
    Format a task prompt using the specified template.
    
    Args:
        user_input: The user's natural language input
        template_type: Either "task_parse" or "fallback"
        error_context: Error information for fallback template
        **kwargs: Additional template variables
        
    Returns:
        Formatted prompt string ready for LLM processing
    """
    if template_type == "task_parse":
        return TASK_PARSE_TEMPLATE.format(user_input=user_input, **kwargs)
    elif template_type == "fallback":
        return FALLBACK_TEMPLATE.format(
            user_input=user_input,
            error_context=error_context or "Unknown error",
            **kwargs
        )
    else:
        raise ValueError(f"Unknown template type: {template_type}")