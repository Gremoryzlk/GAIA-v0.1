-- GAIA v7.3 SQLite WAL Schema
-- Contains exactly 3 tables for core memory persistence

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    agent_type TEXT NOT NULL,
    level INTEGER NOT NULL,
    status TEXT DEFAULT 'IDLE' CHECK(status IN ('IDLE', 'BUSY', 'ERROR', 'ISOLATED')),
    rogue_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Working Memory: Short-term task context (TTL 24h, LRU eviction)
CREATE TABLE IF NOT EXISTS working_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    signature TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttl_hours INTEGER DEFAULT 24,
    tags TEXT,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_working_memory_agent_key ON working_memory(agent_id, key);
CREATE INDEX IF NOT EXISTS idx_working_memory_agent ON working_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_working_memory_key ON working_memory(key);
CREATE INDEX IF NOT EXISTS idx_working_memory_created ON working_memory(created_at);

-- Long-Term Memory: Facts and events with confidence scores
CREATE TABLE IF NOT EXISTS longterm_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    signature TEXT,
    confidence_score REAL DEFAULT 1.0,
    ttl TIMESTAMP,
    provenance_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_longterm_memory_agent_key ON longterm_memory(agent_id, key);
CREATE INDEX IF NOT EXISTS idx_longterm_memory_agent ON longterm_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_longterm_memory_key ON longterm_memory(key);
CREATE INDEX IF NOT EXISTS idx_longterm_memory_confidence ON longterm_memory(confidence_score);
CREATE INDEX IF NOT EXISTS idx_longterm_memory_ttl ON longterm_memory(ttl);

-- Decision Memory: All decisions with traces for GRPO training
CREATE TABLE IF NOT EXISTS decision_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    trace TEXT NOT NULL,
    signature TEXT,
    reward REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_decision_memory_agent ON decision_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_decision_memory_created ON decision_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_decision_memory_reward ON decision_memory(reward);