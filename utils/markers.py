
def asses_ci(cii):
    if cii > 1.5:
        return 0 # unstable
    if cii > 0.7:
        return 1 # lower stability
    return 2 # stable