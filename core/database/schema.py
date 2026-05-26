"""SQLite database schema for GAIA v7.3."""

import sqlite3
from pathlib import Path
from typing import Optional


def init_database(db_path: str = "data/system.db") -> sqlite3.Connection:
    """Initialize SQLite database with GAIA v7.3 schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    
    cursor = conn.cursor()
    
    # Working memory table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS working_memory (
            key TEXT PRIMARY KEY,
            value TEXT,
            tags TEXT,
            ttl INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            provenance_hash TEXT,
            signature TEXT
        )
    """)
    
    # Long-term memory table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS longterm_memory (
            key TEXT PRIMARY KEY,
            value TEXT,
            tags TEXT,
            confidence_score REAL DEFAULT 1.0,
            ttl INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            provenance_hash TEXT,
            signature TEXT
        )
    """)
    
    # Decision memory table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decision_memory (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id TEXT,
            confidence REAL,
            inputs TEXT,
            outputs TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            trace_steps TEXT,
            signature TEXT
        )
    """)
    
    conn.commit()
    return conn


def get_connection(db_path: str = "data/system.db") -> sqlite3.Connection:
    """Get database connection with WAL mode."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    return conn