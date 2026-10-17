"""Single economic convention: extraction in period 0 has no discount."""


def discounted_value(value: float, period: int, discount_rate: float) -> float:
    return value / (1 + discount_rate) ** period


def extraction_reward(value: float, period: int, discount_rate: float) -> float:
    return discounted_value(value, period, discount_rate)
