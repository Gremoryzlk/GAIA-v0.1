"""HMAC handler for GAIA v7.3 event signing."""

import hashlib
import hmac
from typing import Optional


class HMACHandler:
    """HMAC-SHA256 handler for event signing."""
    
    def __init__(self, hmac_key: bytes, stream=None):
        """Initialize HMACHandler with key."""
        self._key = hmac_key
        self._stream = stream
    
    def sign(self, data: str) -> str:
        """Sign data with HMAC-SHA256."""
        return hmac.new(
            self._key,
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def verify(self, data: str, signature: str) -> bool:
        """Verify signature against data."""
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)