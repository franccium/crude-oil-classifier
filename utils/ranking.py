

def light_oil_ans(m_s_value, m_tsi):
    tsi_effectiveness = 0.7
    s_value_effectiveness = 0.6
    summed_effect = tsi_effectiveness + s_value_effectiveness

    return give_verdict((tsi_effectiveness * m_tsi + s_value_effectiveness * m_s_value) /
                        summed_effect)

def medium_oil_ans(m_cii, m_p_value, m_s_value):
    p_value_effectiveness = 0.95
    s_value_effectiveness = 0.85
    sara_effectiveness = 0.65
    summed_effect = p_value_effectiveness + s_value_effectiveness + sara_effectiveness

    return give_verdict((p_value_effectiveness * m_p_value + s_value_effectiveness * m_s_value
                        + sara_effectiveness * m_cii) / summed_effect)

def heavy_oil_ans(m_cii, m_p_value, m_s_value):
    p_value_effectiveness = 1.0
    s_value_effectiveness = 0.905
    sara_effectiveness = 0.762
    summed_effect = p_value_effectiveness + s_value_effectiveness + sara_effectiveness

    return give_verdict((p_value_effectiveness * m_p_value + s_value_effectiveness * m_s_value
                        + m_cii * sara_effectiveness ) / summed_effect)


def give_verdict(val):
    if val > 1.33:
        return 'stable'
    if val > 0.66:
        return 'reduced stability'
    return 'unstable'
