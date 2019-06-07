import model


def test_model():
    m = model.get_model()
    assert isinstance(m, dict)
    assert isinstance(m['Bonjour le monde!'], str)
