"""Runtime SSL bootstrap helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TRUSTSTORE_READY: bool | None = None


def initialize_ssl_truststore_once() -> bool:
    """Inject system trust store into Python SSL once per process.

    Returns:
        True when truststore was successfully injected and should be used.
        False when truststore is unavailable or injection failed.
    """
    global _TRUSTSTORE_READY

    if _TRUSTSTORE_READY is not None:
        return _TRUSTSTORE_READY

    try:
        import truststore
    except Exception:
        _TRUSTSTORE_READY = False
        return False

    try:
        truststore.inject_into_ssl()
        _TRUSTSTORE_READY = True
        logger.info("Initialized truststore SSL integration")
        return True
    except Exception as exc:
        _TRUSTSTORE_READY = False
        logger.warning("Failed to initialize truststore SSL integration: %s", exc)
        return False
