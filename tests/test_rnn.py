import pytest
import torch
from strategic_classification.models.rnn.rnn import MyRNN
import strategic_classification.models.rnn.rnn as rnn


class DummyCCP:
    def __init__(self, x_dim, h_dim):
        pass

    def optimize_X(self, x, H, W_hy, W_hh, W_xh, b, slope):
        return x

class DummyDelta:
    def __init__(self, x_dim):
        pass
    
    def optimize_X(self, x, H, W_hy, W_hh, W_xh, b, F_DER):
        return x

@pytest.fixture(autouse=True)
def patch_rnnccp_and_delta(monkeypatch):
    monkeypatch.setattr(rnn, "RNNCCP", DummyCCP)
    monkeypatch.setattr(rnn, "DeltaRNN", DummyDelta)

@pytest.fixture
def dummy_dims():
    x_dim = 8
    h_dim = 4
    seq_len = 5
    batch = 3
    return x_dim, h_dim, seq_len, batch

@pytest.fixture
def dummy_input(dummy_dims):
    x_dim, h_dim, seq_len, batch = dummy_dims
    X = torch.randn(batch, seq_len, x_dim, dtype=torch.float64)
    Y = torch.randint(0, 2, (batch, 1), dtype=torch.float64) * 2 - 1
    return X, Y

def test_init_shapes(dummy_dims):
    x_dim, h_dim, _, _ = dummy_dims
    model = MyRNN(x_dim, h_dim, 1.0, 1.0)
    assert model.W_hh.shape == (h_dim, h_dim)
    assert model.W_xh.shape == (h_dim, x_dim)
    assert model.W_hy.shape == (h_dim,)
    assert model.b.shape == (h_dim,)

def test_forward_output_shape(dummy_input, dummy_dims):
    x_dim, h_dim, seq_len, batch = dummy_dims
    model = MyRNN(x_dim, h_dim, 1.0, 1.0)
    X, _ = dummy_input
    out = model.forward(X)
    assert out.shape == (batch,)

def test_loss_scalar(dummy_input, dummy_dims):
    x_dim, h_dim, _, _ = dummy_dims
    model = MyRNN(x_dim, h_dim, 1.0, 1.0)
    X, Y = dummy_input
    Y_pred = torch.randn_like(Y)
    loss = model.loss(Y, Y_pred)
    assert loss.shape == ()
    assert isinstance(loss.item(), float)

def test_calc_accuracy_perfect(dummy_dims):
    x_dim, h_dim, seq_len, batch = dummy_dims
    model = MyRNN(x_dim, h_dim, 1.0, 1.0)
    Y = torch.tensor([[1.], [-1.], [1.]])
    Y_pred = torch.tensor([[2.], [-3.], [0.5]])
    acc = model.calc_accuracy(Y, Y_pred)
    assert acc == 1.0

def test_calc_accuracy_half(dummy_dims):
    x_dim, h_dim, seq_len, batch = dummy_dims
    model = MyRNN(x_dim, h_dim, 1.0, 1.0)
    Y = torch.tensor([[1.], [-1.], [1.], [-1.]])
    Y_pred = torch.tensor([[2.], [3.], [-2.], [-3.]])
    acc = model.calc_accuracy(Y, Y_pred)
    assert acc == 0.5

def test_optimize_X_shape(dummy_input, dummy_dims):
    x_dim, h_dim, seq_len, batch = dummy_dims
    model = MyRNN(x_dim, h_dim, 1.0, 1.0)
    X, _ = dummy_input
    X_opt = model.optimize_X(X)
    assert X_opt.shape == X.shape
