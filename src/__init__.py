"""Triagem de chamados com uma LLM local e GPU obrigatória."""
import os
from pathlib import Path

# Definir antes de importar Hugging Face/PEFT: essas bibliotecas leem o cache no import.
_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(_root / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
