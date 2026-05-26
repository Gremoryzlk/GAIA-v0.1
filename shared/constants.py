"""
GAIA v7.3 System Constants

This module contains all system-wide constants used across the GAIA platform.
Constants are organized by functional area and maintained in alphabetical order
within each group for easy maintenance and discovery.
"""

# =============================================================================
# Agent Configuration
# =============================================================================
MAX_AGENTS_WITHOUT_APPROVAL = 10

# =============================================================================
# Decision Thresholds
# =============================================================================
AUTO_EXECUTE_THRESHOLD = 0.8
ASK_USER_THRESHOLD = 0.5
NOTIFY_THRESHOLD = 0.5

# =============================================================================
# Circuit Breaker Configuration
# =============================================================================
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 0.2
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SEC = 60

# =============================================================================
# CPU Limits
# =============================================================================
CPU_HARD_LIMIT_PCT = 85
CPU_SOFT_LIMIT_PCT = 70

# =============================================================================
# EventBus Configuration
# =============================================================================
EVENTBUS_QUEUE_SIZE = 1000
RATE_LIMIT_PER_AGENT = 10
BACKPRESSURE_THRESHOLD = 0.85

# =============================================================================
# Evolution Configuration
# =============================================================================
GROWTH_ENGINE_VARIANTS = 5
MAX_ROLLBACK_DEPTH = 5
NUM_VARIANTS = 8

# =============================================================================
# Security Configuration
# =============================================================================
HMAC_KEY_LENGTH = 32
ROGUE_SCORE_ISOLATE = 5
SHADOW_DIVERGENCE_THRESHOLD = 0.05

# =============================================================================
# Memory Configuration
# =============================================================================
IDLE_PURGE_MINUTES = 15
MEMORY_LONGTERM_MAX_ENTRIES = 1000000
MEMORY_WORKING_TTL_HOURS = 24

# =============================================================================
# RAM Limits
# =============================================================================
MAX_RAM_MB = 14336
RAM_HARD_LIMIT_GB = 13.5
RAM_SOFT_LIMIT_GB = 12.0

# =============================================================================
# Tool Event Routing Constants
# =============================================================================
MAX_TOOL_RETRIES = 3
TOOL_REQUEST_EVENT = "tool.request"
TOOL_RESULT_EVENT = "tool.result"
TOOL_TIMEOUT_SECONDS = 30

# =============================================================================
# Watchdog Configuration
# =============================================================================
WATCHDOG_HANG_SEC = 30
# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_CACHE_TTL_HOURS: int = 24
LLM_MAX_TOKENS: int = 4096
LLM_TIMEOUT_SECONDS: int = 60
LLM_MODEL_DEFAULT: str = "phi3-mini"

# ── Tools / Hephaestus ────────────────────────────────────────────────────────
TOOL_WORKSPACE_DIR: str = "modules/hephaestus/data/workspace"
TOOL_MAX_FILE_SIZE_BYTES: int = 100 * 1024
TOOL_MAX_MEMORY_RESULTS: int = 100

# ── Titan ──────────────────────────────────────────────────────────────────────
TITAN_TIMEOUT_SECONDS: int = 240
TITAN_JWT_EXPIRY_SECONDS: int = 3600
TITAN_MAX_SERVERS: int = 10
TITAN_RESULT_MEMORY_MAX_BYTES: int = 1 * 1024 * 1024      # 1MB
TITAN_RESULT_FILE_MAX_BYTES: int = 100 * 1024 * 1024     # 100MB
TITAN_CHECKPOINT_INTERVAL_STEPS: int = 10
