# Repository Structure

```
beyond_transformer/
├── README.md                    # Main documentation
├── SETUP.md                     # Installation guide
├── LICENSE                      # MIT License
├── setup.py                     # Package setup
├── pyproject.toml              # Project config
├── requirements_maas.txt       # MaaS dependencies
├── test_maas_simple.py         # Quick test script
│
├── src/pulse/                  # Core PULSE implementation
│   ├── core/                   # Core components
│   │   ├── attention.py
│   │   ├── memory.py
│   │   ├── state.py
│   │   └── ...
│   ├── models/                 # Model implementations
│   │   └── pulse.py
│   ├── utils/                  # Utilities
│   └── maas/                   # MaaS integration ⭐
│       ├── memory_service.py   # Core memory service
│       ├── api.py              # REST API
│       ├── consolidation.py    # Consolidation engine
│       ├── query_engine.py     # Query processing
│       └── server.py           # Flask server
│
├── docs/                       # Documentation
│   ├── MAAS.md                # MaaS guide
│   └── ARCHITECTURE.md        # Architecture details
│
├── examples/                   # Usage examples
│   ├── README.md
│   ├── maas_basic_usage.py
│   ├── maas_advanced_usage.py
│   ├── maas_integration_demo.py
│   └── maas_server_example.py
│
├── tests/                      # Test suite
│   ├── README.md
│   └── test_maas.py
│
├── scripts/                    # Utility scripts
│   └── train.py
│
└── configs/                    # Configuration files
    ├── pulse_base.yaml
    └── pulse_small.yaml
```

## Key Files

- **`README.md`** - Start here
- **`SETUP.md`** - Installation instructions
- **`docs/MAAS.md`** - MaaS documentation
- **`test_maas_simple.py`** - Quick test
- **`examples/`** - Usage examples
