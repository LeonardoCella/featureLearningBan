def sherman_morrison(m_inv, x):
    x = x.reshape((-1,1))
    minv = m_inv
    minv -= m_inv.dot(x.dot(x.T.dot(m_inv))) / ( 1 + x.T.dot(m_inv.dot(x)))
    return minv
