# Repository Structure

## Overview

```
beyond_transformer/
├── README.md                    # Main documentation
├── SETUP.md                     # Installation guide
├── CONTRIBUTING.md              # Development guide
├── LICENSE                      # MIT License
│
├── src/pulse/                   # Core implementation
│   ├── core/                    # Core PULSE components
│   ├── models/                  # Model implementations
│   ├── utils/                   # Utilities
│   └── maas/                    # MaaS integration ⭐
│
├── docs/                        # Documentation
│   ├── README.md                # Documentation index
│   ├── MAAS.md                  # MaaS guide
│   ├── ARCHITECTURE.md          # Architecture details
│   └── STRUCTURE.md             # This file
│
├── examples/                    # Usage examples
│   ├── README.md
│   ├── maas_basic_usage.py
│   ├── maas_advanced_usage.py
│   ├── maas_integration_demo.py
│   └── maas_server_example.py
│
├── tests/                       # Test suite
│   ├── README.md
│   └── test_maas.py
│
├── scripts/                     # Utility scripts
├── configs/                     # Configuration files
├── setup.py                     # Package setup
├── pyproject.toml              # Project config
├── requirements_maas.txt       # MaaS dependencies
└── test_maas_simple.py         # Quick test script
```

## Key Directories

### `src/pulse/`
Core PULSE implementation and MaaS integration.

### `src/pulse/maas/`
Memory-as-a-Service components:
- `memory_service.py` - Core memory service
- `api.py` - REST API interface
- `consolidation.py` - Consolidation engine
- `query_engine.py` - Query processing
- `server.py` - Flask server

### `docs/`
All documentation files.

### `examples/`
Working code examples for MaaS.

### `tests/`
Test suite for all components.

## Quick Navigation

- **Getting Started**: `README.md` → `SETUP.md`
- **MaaS Documentation**: `docs/MAAS.md`
- **Examples**: `examples/`
- **Tests**: `test_maas_simple.py` or `tests/`
- **Architecture**: `docs/ARCHITECTURE.md`
