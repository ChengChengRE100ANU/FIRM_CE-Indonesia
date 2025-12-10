from typing import List


def parse_comma_separated(value: str) -> List[str]:
    """
    Parse a comma-separated string into a list of trimmed, non-empty strings.

    Parameters:
    -------
    value (str): A string containing comma-separated values.

    Returns:
    -------
    List[str]: A list of cleaned strings with whitespace removed and empty entries excluded.
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_divide(num: float, denom: float) -> float:
    """Safe division for calculating levelised costs when total dispatch energy from the asset is 0."""
    return num / denom if denom != 0 else 0.0
