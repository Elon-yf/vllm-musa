from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
custom_ar = pytest.importorskip(
    "vllm_musa.distributed.device_communicators.musa_jit_custom_all_reduce"
)


def test_register_graph_buffers_populates_persistent_rank_data(monkeypatch):
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl.rank = 0
    impl.world_size = 2
    impl.group = object()
    impl.device = torch.device("cpu")
    impl.graph_rank_data = torch.zeros((4, 8), dtype=torch.int64)
    impl._pending_graph_inputs = [torch.empty(4), torch.empty(8)]
    impl._graph_input_refs = []
    impl._graph_peer_bases = {}
    impl._graph_opened_ptrs = []
    impl._next_graph_slot = 0

    local_handles = [(b"a" * 64, 8), (b"b" * 64, 16)]
    peer_handles = [(b"c" * 64, 24), (b"d" * 64, 32)]
    monkeypatch.setattr(impl, "_graph_pointer_meta", lambda tensor: local_handles.pop(0))

    def all_gather_object(output, local_meta, group):
        assert group is impl.group
        output[0] = list(local_meta)
        output[1] = peer_handles

    peer_bases = {ord("c"): 10_000, ord("d"): 20_000}

    class FakeRuntime:
        def ipc_open_mem_handle(self, handle):
            first_byte = ctypes.string_at(ctypes.byref(handle), 1)[0]
            return ctypes.c_void_p(peer_bases[first_byte])

    monkeypatch.setattr(custom_ar.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(custom_ar, "_MusaRTLibrary", FakeRuntime)
    monkeypatch.setattr(
        custom_ar.torch,
        "musa",
        SimpleNamespace(synchronize=lambda device: None),
    )

    inputs = list(impl._pending_graph_inputs)
    impl._register_graph_buffers()

    assert impl.graph_rank_data[0, 0].item() == inputs[0].data_ptr()
    assert impl.graph_rank_data[0, 1].item() == 10_024
    assert impl.graph_rank_data[1, 0].item() == inputs[1].data_ptr()
    assert impl.graph_rank_data[1, 1].item() == 20_032
    assert impl._next_graph_slot == 2
    assert impl._pending_graph_inputs == []
    assert impl._graph_input_refs == inputs
    assert sorted(impl._graph_opened_ptrs) == [10_000, 20_000]


def test_capture_resets_state_when_registration_fails(monkeypatch):
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl._IS_CAPTURING = False
    impl._pending_graph_inputs = []

    def fail_registration():
        raise RuntimeError("registration failed")

    monkeypatch.setattr(impl, "_register_graph_buffers", fail_registration)
    with pytest.raises(RuntimeError, match="registration failed"):
        with impl.capture():
            assert impl._IS_CAPTURING is True
            impl._pending_graph_inputs.append(torch.empty(1))
    assert impl._IS_CAPTURING is False


def test_graph_launch_uses_next_persistent_slot(monkeypatch):
    impl = object.__new__(custom_ar._MusaJitCustomAllreduceImpl)
    impl.graph_rank_data = torch.zeros((8, 8), dtype=torch.int64)
    impl.signal_ptrs_cpu = torch.zeros(2, dtype=torch.int64)
    impl._pending_graph_inputs = []
    impl._next_graph_slot = 3
    impl.rank = 0
    impl.world_size = 2
    captured = {}

    def launch(rank_data, signals, input_tensor, output, rank, world_size, shot):
        captured["rank_data"] = rank_data
        captured["signals"] = signals
        captured["input"] = input_tensor
        captured["output"] = output
        captured["rank"] = rank
        captured["world_size"] = world_size
        captured["shot"] = shot

    monkeypatch.setattr(custom_ar.jit_ar, "launch_graph_registered", launch)
    monkeypatch.setattr(custom_ar.jit_ar, "preferred_shot", lambda world, nbytes: 2)

    input_tensor = torch.randn(4, dtype=torch.bfloat16)
    output = impl._graph_custom_all_reduce_impl(input_tensor)

    assert captured["rank_data"].data_ptr() == impl.graph_rank_data[3].data_ptr()
    assert captured["input"] is input_tensor
    assert captured["output"] is output
    assert captured["rank"] == 0
    assert captured["world_size"] == 2
    assert captured["shot"] == 2
    assert impl._pending_graph_inputs == [input_tensor]
