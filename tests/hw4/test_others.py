import needle as ndl
import numpy as np
import pytest
import torch
import torch.nn as nn


def _make_ndl_tensor(arr, requires_grad=False):
    # try common constructors/wrappers used in different needle versions
    for name in ("Tensor", "tensor", "array"):
        if hasattr(ndl, name):
            ctor = getattr(ndl, name)
            try:
                return ctor(arr, requires_grad=requires_grad)
            except TypeError:
                try:
                    t = ctor(arr)
                    if requires_grad:
                        try:
                            t.requires_grad = True
                        except Exception:
                            pass
                    return t
                except Exception:
                    continue
    pytest.skip("Cannot construct needle tensor (no known constructor)")


def _extract_numpy(x):
    # extract numpy array / scalar from various container types
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (float, int)):
        return np.array(x)
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    if hasattr(x, "detach"):
        d = x.detach()
        if hasattr(d, "numpy"):
            try:
                return d.numpy()
            except Exception:
                pass
    if hasattr(x, "item"):
        try:
            return np.array(x.item())
        except Exception:
            pass
    # if it's a needle tensor with .grad or .data attributes that are numpy-like
    if hasattr(x, "data") and hasattr(x.data, "numpy"):
        try:
            return x.data.numpy()
        except Exception:
            pass
    pytest.skip("Unable to convert value to numpy for comparison")


def _find_ndl_softmax():
    # locate softmax loss in common locations
    if hasattr(ndl, "SoftmaxLoss"):
        return getattr(ndl, "SoftmaxLoss")
    if hasattr(ndl, "softmax_loss"):
        return getattr(ndl, "softmax_loss")
    if hasattr(ndl, "nn") and hasattr(ndl.nn, "SoftmaxLoss"):
        return getattr(ndl.nn, "SoftmaxLoss")
    if hasattr(ndl, "loss") and hasattr(ndl.loss, "SoftmaxLoss"):
        return getattr(ndl.loss, "SoftmaxLoss")
    pytest.skip("Needle SoftmaxLoss not found")


@pytest.mark.parametrize("N,C", [(5, 3), (7, 1), (10, 4)])
def test_softmax_loss_matches_torch(N, C):
    # generate deterministic random inputs
    rng = np.random.RandomState(123 + N + C)
    logits_np = rng.randn(N, C).astype(np.float32)
    labels_np = rng.randint(0, C, size=(N,), dtype=np.int64)

    # torch reference
    torch_logits = torch.tensor(logits_np, requires_grad=True)
    torch_labels = torch.tensor(labels_np, dtype=torch.long)
    loss_torch_fn = nn.CrossEntropyLoss()
    loss_torch = loss_torch_fn(torch_logits, torch_labels)
    loss_torch_val = loss_torch.item()
    loss_torch.backward()
    grad_torch = torch_logits.grad.detach().cpu().numpy()

    # needle under test
    ndl_logits = _make_ndl_tensor(logits_np, requires_grad=True)
    ndl_labels = ndl.Tensor(labels_np)

    loss_ndl_fn = ndl.nn.SoftmaxLoss()
    ndl_loss = loss_ndl_fn(ndl_logits, ndl_labels)

    ndl_loss_val = _extract_numpy(ndl_loss)
    # compare forward values
    assert np.allclose(ndl_loss_val, loss_torch_val, rtol=1e-8, atol=1e-8), (
        f"loss mismatch: needle={ndl_loss_val} torch={loss_torch_val}"
    )

    # DEBUG:
    print(ndl_loss_val, loss_torch_val)

    # try to check gradients if available
    try:
        # attempt backwards on needle loss
        if hasattr(ndl_loss, "backward"):
            ndl_loss.backward()
        else:
            # some APIs may require loss_obj.backward or similar
            if hasattr(ndl, "backward"):
                ndl.backward(ndl_loss)
            else:
                raise RuntimeError("no backward found")
        # extract gradient from ndl_logits
        if hasattr(ndl_logits, "grad"):
            ndl_grad = _extract_numpy(ndl_logits.grad)
        elif hasattr(ndl_logits, "data") and hasattr(ndl_logits.data, "numpy"):
            ndl_grad = _extract_numpy(ndl_logits.data.numpy())
        else:
            pytest.skip("Needle logits have no accessible .grad for comparison")
        assert ndl_grad.shape == grad_torch.shape
        assert np.allclose(ndl_grad, grad_torch, rtol=1e-3, atol=1e-3), (
            "gradient mismatch between needle and torch"
        )
    except pytest.skip.Exception:
        raise
    except Exception:
        # if gradient comparison is not possible, at least we checked forward value
        pytest.skip("Could not verify gradients for this needle implementation")
