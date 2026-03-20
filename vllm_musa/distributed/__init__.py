try:
    from mooncake.engine import TransferEngine

    import vllm_musa.distributed.kv_transfer.kv_connector.v1.mooncake_connector
except ImportError:
    pass
