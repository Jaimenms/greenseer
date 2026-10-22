"""Gera (e opcionalmente executa) os notebooks a partir das fontes em scripts/notebooks_src.

Formato das fontes (Python válido):

    # %% [markdown]
    \"\"\"
    texto em Markdown
    \"\"\"

    # %%
    código

Uso:
    uv run python scripts/build_notebooks.py            # só gera os .ipynb
    uv run python scripts/build_notebooks.py --execute  # gera e executa (grava as saídas)
    uv run python scripts/build_notebooks.py --execute 02   # apenas os que começam com "02"
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RAIZ = Path(__file__).resolve().parents[1]
FONTES = RAIZ / "scripts" / "notebooks_src"
DESTINO = RAIZ / "notebooks"


def parse(texto: str):
    partes = re.split(r"^# %%(.*)$", texto, flags=re.MULTILINE)
    # partes = [prefixo, marcador, corpo, marcador, corpo, ...]
    celulas = []
    for i in range(1, len(partes), 2):
        marcador, corpo = partes[i].strip(), partes[i + 1]
        corpo = corpo.strip("\n")
        if "[markdown]" in marcador:
            corpo = corpo.strip()
            if corpo.startswith('"""') and corpo.endswith('"""'):
                corpo = corpo[3:-3].strip("\n")
            celulas.append(new_markdown_cell(corpo))
        else:
            if corpo.strip():
                celulas.append(new_code_cell(corpo))
    return celulas


def gerar(fonte: Path) -> Path:
    nb = new_notebook(cells=parse(fonte.read_text(encoding="utf-8")))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (greenseer)", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    destino = DESTINO / (fonte.stem + ".ipynb")
    nbformat.write(nb, destino)
    return destino


def executar(caminho: Path) -> None:
    from nbconvert.preprocessors import ExecutePreprocessor

    nb = nbformat.read(caminho, as_version=4)
    ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
    t0 = time.time()
    ep.preprocess(nb, {"metadata": {"path": str(caminho.parent)}})
    nbformat.write(nb, caminho)
    print(f"  executado em {time.time() - t0:.0f}s")


def main(argv):
    executar_flag = "--execute" in argv
    filtros = [a for a in argv if not a.startswith("--")]
    DESTINO.mkdir(exist_ok=True)
    for fonte in sorted(FONTES.glob("*.py")):
        if filtros and not any(fonte.name.startswith(f) for f in filtros):
            continue
        destino = gerar(fonte)
        print(f"gerado {destino.relative_to(RAIZ)}")
        if executar_flag:
            executar(destino)


if __name__ == "__main__":
    main(sys.argv[1:])
