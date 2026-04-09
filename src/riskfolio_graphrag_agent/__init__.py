"""Top-level package for the Riskfolio GraphRAG and KG induction system.

The package contains both the temporary legacy graph path and the redesigned
staged KG induction pipeline. Import concrete pipeline modules directly when
you need the redesigned graph stack in order to keep package import side
effects minimal.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
