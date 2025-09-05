import builtins
import importlib
import sys

from scperturb_cmap.utils.device import get_device


def test_get_device_returns_known_value():
    device = get_device()
    assert device in {"cuda", "mps", "cpu"}


def test_get_device_cuda_branch(monkeypatch):
    class DummyCUDA:
        @staticmethod
        def is_available():
            return True

    class DummyMPS:
        @staticmethod
        def is_available():
            return False

    class DummyBackends:
        mps = DummyMPS()

    class DummyTorch:
        cuda = DummyCUDA()
        backends = DummyBackends()

    monkeypatch.setitem(sys.modules, "torch", DummyTorch)
    import scperturb_cmap.utils.device as dev

    importlib.reload(dev)
    assert dev.get_device() == "cuda"


def test_get_device_mps_branch(monkeypatch):
    class DummyCUDA:
        @staticmethod
        def is_available():
            return False

    class DummyMPS:
        @staticmethod
        def is_available():
            return True

    class DummyBackends:
        mps = DummyMPS()

    class DummyTorch:
        cuda = DummyCUDA()
        backends = DummyBackends()

    monkeypatch.setitem(sys.modules, "torch", DummyTorch)
    import scperturb_cmap.utils.device as dev

    importlib.reload(dev)
    assert dev.get_device() == "mps"


def test_get_device_cpu_branch_when_none_available(monkeypatch):
    class DummyCUDA:
        @staticmethod
        def is_available():
            return False

    class DummyMPS:
        @staticmethod
        def is_available():
            return False

    class DummyBackends:
        mps = DummyMPS()

    class DummyTorch:
        cuda = DummyCUDA()
        backends = DummyBackends()

    monkeypatch.setitem(sys.modules, "torch", DummyTorch)
    import scperturb_cmap.utils.device as dev

    importlib.reload(dev)
    assert dev.get_device() == "cpu"


def test_get_device_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Direct call uses local import inside the function
    assert get_device() == "cpu"
