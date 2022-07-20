def test_transforers():
    import transformers as package
    assert str(package.__version__) == "4.16.2"

def test_torch():
    import torch as package
    assert str(package.__version__) == "1.11.0+cpu"
