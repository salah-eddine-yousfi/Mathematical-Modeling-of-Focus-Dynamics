from __future__ import annotations
from typing import Optional

CONCENTRATED_CLASSES = {"focused_writing", "focused_reading"}
NON_CONCENTRATED_CLASSES = {"not_activity", "not_phone"}


def map_class_to_binary_or_none(class_name: str) -> Optional[int]:
    """
    Retour:
      1  -> concentré
      0  -> non concentré
      None -> unknown / incertain
    """
    if class_name in CONCENTRATED_CLASSES:
        return 1
    if class_name in NON_CONCENTRATED_CLASSES:
        return 0
    return None
