from __future__ import annotations
from dataclasses import dataclass

import dolfin


@dataclass
class ECG:
    extracellular_potential: list[dolfin.Function]
