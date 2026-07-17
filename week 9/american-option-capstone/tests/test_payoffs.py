from src.pricing.payoffs import put_payoff, call_payoff


def test_put_payoff_itm():
    assert put_payoff(80, 100) == 20


def test_put_payoff_otm():
    assert put_payoff(120, 100) == 0


def test_call_payoff_itm():
    assert call_payoff(120, 100) == 20


def test_call_payoff_otm():
    assert call_payoff(80, 100) == 0


def test_payoffs_never_negative():
    assert put_payoff(1000, 100) >= 0
    assert call_payoff(1, 100) >= 0
