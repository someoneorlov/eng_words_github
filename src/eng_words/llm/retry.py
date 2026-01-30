"""Universal retry and validation utilities for LLM calls.

Provides robust retry logic and JSON validation for any LLM provider.
"""

import json
import logging
import time
from typing import Any, Callable, TypeVar

from eng_words.llm.base import LLMProvider, LLMResponse
from eng_words.llm.response_cache import ResponseCache

logger = logging.getLogger(__name__)

T = TypeVar("T")


def call_llm_with_retry(
    provider: LLMProvider,
    prompt: str,
    cache: ResponseCache | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    validate_json: bool = False,
    json_schema: dict | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> LLMResponse:
    """Call LLM with retry logic and optional JSON validation.
    
    Handles:
    - Network errors
    - Rate limiting
    - JSON parsing errors (if validate_json=True)
    - Caching (if cache provided)
    
    Args:
        provider: LLM provider instance
        prompt: Prompt text
        cache: Optional response cache
        max_retries: Maximum number of retries (default: 3)
        retry_delay: Base delay between retries in seconds (default: 1.0)
        validate_json: If True, validates response as JSON
        json_schema: Optional JSON schema for validation
        on_retry: Optional callback called on each retry (attempt_num, error)
        
    Returns:
        LLMResponse from the provider
        
    Raises:
        ValueError: If JSON validation fails after all retries
        Exception: If LLM call fails after all retries
    """
    # Check cache first
    if cache:
        model_name = getattr(provider, "model", "unknown-model")
        temperature = getattr(provider, "temperature", 0.0)
        cache_key = cache.generate_key(model_name, prompt, temperature)
        cached_response = cache.get(cache_key)
        
        if cached_response:
            logger.debug(f"Cache hit for LLM call")
            if validate_json:
                try:
                    _validate_json_response(cached_response.content, json_schema)
                except ValueError as e:
                    logger.warning(f"Cached response failed JSON validation: {e}, retrying")
                    # Continue to LLM call
            else:
                return cached_response
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # Call LLM
            if validate_json:
                # Try complete_json first if available
                if hasattr(provider, "complete_json"):
                    try:
                        result_dict = provider.complete_json(prompt, schema=json_schema)
                        # complete_json internally calls complete(), so we need to get the response
                        # Call complete() to get full LLMResponse with tokens
                        response = provider.complete(prompt)
                        # But use the parsed JSON content (re-encode to ensure it's valid JSON)
                        response.content = json.dumps(result_dict, ensure_ascii=False)
                        result = response
                    except (ValueError, json.JSONDecodeError) as e:
                        # complete_json failed to parse - fall back to complete() and validate
                        logger.warning(f"complete_json failed: {e}, falling back to complete()")
                        response = provider.complete(prompt)
                        _validate_json_response(response.content, json_schema)
                        result = response
                    except Exception as e:
                        # Other errors from complete_json - just use complete()
                        logger.warning(f"complete_json error: {e}, falling back to complete()")
                        response = provider.complete(prompt)
                        _validate_json_response(response.content, json_schema)
                        result = response
                else:
                    response = provider.complete(prompt)
                    _validate_json_response(response.content, json_schema)
                    result = response
            else:
                result = provider.complete(prompt)
            
            # Cache the response
            if cache:
                try:
                    cache.set(cache_key, result)
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
            
            return result
            
        except (ValueError, json.JSONDecodeError) as e:
            # JSON validation error
            if not validate_json:
                # Unexpected JSON error when not validating
                raise
            
            last_error = e
            logger.warning(
                f"JSON validation failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
            )
            
            if on_retry:
                on_retry(attempt + 1, e)
            
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying LLM call in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                # Last attempt failed - raise exception
                last_error = e
                raise ValueError(f"JSON validation failed after {max_retries + 1} attempts: {e}")
                
        except Exception as e:
            # Other errors (network, rate limit, etc.)
            last_error = e
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
            )
            
            if on_retry:
                on_retry(attempt + 1, e)
            
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying LLM call in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                # Last attempt failed - raise exception
                last_error = e
                raise Exception(f"LLM call failed after {max_retries + 1} attempts: {e}")
    
    # Should never reach here, but just in case
    if last_error:
        raise last_error
    raise Exception("LLM call failed for unknown reason")


def call_llm_json(
    provider: LLMProvider,
    prompt: str,
    cache: ResponseCache | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    json_schema: dict | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> dict[str, Any]:
    """Call LLM and parse JSON response with retry logic.
    
    Convenience wrapper around call_llm_with_retry that:
    - Automatically validates JSON
    - Returns parsed dict instead of LLMResponse
    
    Args:
        provider: LLM provider instance
        prompt: Prompt text
        cache: Optional response cache
        max_retries: Maximum number of retries (default: 3)
        retry_delay: Base delay between retries in seconds (default: 1.0)
        json_schema: Optional JSON schema for validation
        on_retry: Optional callback called on each retry (attempt_num, error)
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If JSON parsing/validation fails after all retries
    """
    response = call_llm_with_retry(
        provider=provider,
        prompt=prompt,
        cache=cache,
        max_retries=max_retries,
        retry_delay=retry_delay,
        validate_json=True,
        json_schema=json_schema,
        on_retry=on_retry,
    )
    
    return _parse_json_response(response.content)


def _validate_json_response(content: str, schema: dict | None = None) -> None:
    """Validate JSON response content.
    
    Args:
        content: Response content to validate
        schema: Optional JSON schema for validation
        
    Raises:
        ValueError: If content is not valid JSON or doesn't match schema
    """
    # Try to parse JSON
    try:
        parsed = _parse_json_response(content)
    except ValueError as e:
        raise ValueError(f"Invalid JSON response: {e}")
    
    # Validate against schema if provided
    if schema:
        # Basic schema validation (can be extended with jsonschema library)
        # For now, just check required fields if specified
        if "required" in schema:
            required_fields = schema["required"]
            missing = [field for field in required_fields if field not in parsed]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse JSON from LLM response content.
    
    Handles:
    - Markdown code blocks (```json ... ```)
    - Trailing commas
    - Incomplete JSON (tries to extract JSON object)
    
    Args:
        content: Raw response content
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If JSON cannot be parsed
    """
    content = content.strip()
    
    # Remove markdown code blocks if present
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    # Try to parse JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Try to extract JSON object from response
        start = content.find("{")
        end = content.rfind("}") + 1
        
        if start >= 0 and end > start:
            try:
                # Try to fix common issues
                json_str = content[start:end]
                
                # Remove trailing commas before closing braces/brackets
                import re
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If all else fails, raise with context
        raise ValueError(f"Failed to parse JSON: {e}. Content preview: {content[:200]}")
