"""Exception types raised by agents and handled by the Orchestrator.

Every exception here is a *known, typed* failure mode -- the Orchestrator catches each of these
(and, as a last resort, any other ``Exception``) and degrades gracefully rather than crashing.
"""


class AgentError(Exception):
    """Base class for all agent-raised errors."""


class PolicyConfigError(AgentError):
    """The policy_terms.json does not have the configuration needed to evaluate this claim
    (e.g. an unknown claim_category). The Orchestrator treats this as unrecoverable for this
    claim and routes to MANUAL_REVIEW."""


class MemberNotFoundError(AgentError):
    """member_id is not present in the policy's member roster. The Orchestrator routes this to
    REJECTED with a message asking the member to verify their member ID."""


class GeminiAPIError(AgentError):
    """The Gemini API call failed (timeout, rate limit, or returned a response that could not be
    validated against the expected schema) after retries were exhausted."""
