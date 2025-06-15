

def light_oil_ans(m_s_value, m_tsi):
    params = [
        (m_tsi, 0.7),
        (m_s_value, 0.6),
    ]
    num, denom = 0, 0
    for val, weight in params:
        if val != -1:
            num += val * weight
            denom += weight
    if denom == 0:
        return 'Unknown'
    return give_verdict(num / denom)

def medium_oil_ans(m_cii, m_p_value, m_s_value):
    params = [
        (m_p_value, 0.95),
        (m_s_value, 0.85),
        (m_cii, 0.65),
    ]
    num, denom = 0, 0
    for val, weight in params:
        if val != -1:
            num += val * weight
            denom += weight
    if denom == 0:
        return 'Unknown'
    return give_verdict(num / denom)

def heavy_oil_ans(m_cii, m_p_value, m_s_value):
    params = [
        (m_p_value, 1.0),
        (m_s_value, 0.905),
        (m_cii, 0.762),
    ]
    num, denom = 0, 0
    for val, weight in params:
        if val != -1:
            num += val * weight
            denom += weight
    if denom == 0:
        return 'Unknown'
    return give_verdict(num / denom)


def give_verdict(val):
    if val > 1.33:
        return 'stable'
    if val > 0.66:
        return 'reduced stability'
    return 'unstable'
