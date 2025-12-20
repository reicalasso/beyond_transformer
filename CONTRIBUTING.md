# Contributing to PULSE

## Development Setup

```bash
# Clone repository
git clone <repository-url>
cd beyond_transformer

# Install in development mode
pip install -e ".[dev]"

# Install MaaS dependencies
pip install -r requirements_maas.txt
```

## Running Tests

```bash
# Quick test
python test_maas_simple.py

# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=pulse --cov-report=html
```

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to public functions
- Keep functions focused and small

## Project Structure

```
src/pulse/
├── core/          # Core PULSE components
├── models/        # Model implementations
├── utils/         # Utilities
└── maas/          # MaaS integration
```

## Adding Features

1. Create feature branch
2. Implement with tests
3. Update documentation
4. Submit pull request

## Documentation

- Update `docs/MAAS.md` for MaaS features
- Update `README.md` for major changes
- Add examples to `examples/` directory
