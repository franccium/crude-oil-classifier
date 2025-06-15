def safe_numeric_input(func):
    def wrapper(value):
        if value == "-":
            return -1
        try:
            value = float(value)
        except (ValueError, TypeError):
            return -1
        return func(value)
    return wrapper

@safe_numeric_input
def asses_cii(cii):
    if cii > 1.5:
        return 0 # unstable
    if cii > 0.7:
        return 1 # lower stability
    return 2 # stable

@safe_numeric_input
def asses_s_value(s_value):
    if s_value > 2.2:
        return 2 # stable
    if s_value > 1.5:
        return 1 # lower stability
    return 0 # unstable

@safe_numeric_input
def asses_p_value(p_value):
    if p_value > 2:
        return 2 # stable
    if p_value > 1.5:
        return 1 # lower stability
    return 0 # unstable

@safe_numeric_input
def asses_tsi(tsi):
    if tsi > 2.5:
        return 2 # stable
    if tsi > 1.5:
        return 1 # lower stability
    return 0 # unstable
