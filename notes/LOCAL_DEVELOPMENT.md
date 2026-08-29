# Local Development Guide

This guide explains how to develop and test `nuwa_sdk` locally against consuming projects (like `nuwa-example`) without publishing to a package registry.

## Overview

When developing new features in `nuwa_sdk`, you need to test them against real projects that depend on it. Since `nuwa_sdk` is installed via Nimble, you need to ensure your local changes are picked up by dependent projects.

## Quick Start

```bash
# 1. Make changes to nuwa-sdk/src/nuwa_sdk.nim
# 2. Run the test script
cd /Users/martin/repos/nuwa/nuwa-sdk
./tests/run_tests.sh
```

This script will:
1. Copy your changes to the example project
2. Build the example project
3. Run integration tests

## Development Workflow

### Step 1: Modify nuwa_sdk

Edit files in `nuwa-sdk/src/`:
- `nuwa_sdk.nim` - Main SDK (withNogil, nuwa_export, numpy wrappers, etc.)

### Step 2: Test Against Consuming Projects

You have three options:

#### Option A: Use the Test Runner Script (Fastest)

```bash
cd /Users/martin/repos/nuwa/nuwa-sdk
./tests/run_tests.sh
```

**Pros:**
- ✅ One command to test everything
- ✅ Handles all copying and building
- ✅ Runs comprehensive tests

**Cons:**
- ❌ Requires being in the nuwa-sdk directory

#### Option B: Manual Testing

```bash
# In the consuming project directory
cd /Users/martin/repos/nuwa/nuwa-example/example_project

# Copy the updated nuwa_sdk.nim to Nimble's cache
find .nimble/pkgs2 -name "nuwa_sdk.nim" \
  -exec cp /Users/martin/repos/nuwa/nuwa-sdk/src/nuwa_sdk.nim {} \;

# Rebuild
nuwa develop

# Test
pytest
```

**Pros:**
- ✅ Full control over the process
- ✅ Can see intermediate results

**Cons:**
- ❌ Manual steps required
- ❌ Easy to forget the copy step

#### Option C: Automated with NPM-style Watch Mode (Advanced)

Create a watch script that automatically rebuilds when files change:

```bash
#!/bin/bash
while true; do
    find /Users/martin/repos/nuwa/nuwa-sdk/src/nuwa_sdk.nim -newer \
        /Users/martin/repos/nuwa/nuwa-example/example_project/.nimble/pkgs2/*/nuwa_sdk.nim 2>/dev/null && do
        echo "nuwa_sdk.nim changed, rebuilding..."
        cd /Users/martin/repos/nuwa/nuwa-example/example_project
        nuwa develop
    done
    sleep 2
done
```

## Testing Your Changes

### Why Python Tests Instead of Nim Tests?

The numpy wrappers require:
1. **Python runtime** - Must have libpython loaded
2. **numpy module** - Must have numpy available
3. **Buffer protocol** - Requires active Python interpreter

Standalone Nim tests can't access these dependencies. The wrappers are designed to be used **from within Nim functions that are called by Python**.

### Running Tests

#### Method 1: Via Test Script

```bash
cd /Users/martin/repos/nuwa/nuwa-sdk
./tests/run_tests.sh
```

#### Method 2: Manual Testing

```bash
# Update nuwa_sdk in example project
cd /Users/martin/repos/nuwa/nuwa-example/example_project
find .nimble/pkgs2 -name "nuwa_sdk.nim" \
  -exec cp /Users/martin/repos/nuwa/nuwa-sdk/src/nuwa_sdk.nim {} \;

# Build
nuwa develop

# Run tests
pytest

# Quick manual test
python3 -c "
import numpy as np
from example_project import numpy_array_sum, numpy_array_sum_fast

arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
assert numpy_array_sum(arr) == 15
assert numpy_array_sum_fast(arr) == 15
print('✓ Tests passed!')
"
```

#### Method 3: Integration Tests

```bash
cd /Users/martin/repos/nuwa/nuwa-example/example_project

# Update nuwa_sdk
find .nimble/pkgs2 -name "nuwa_sdk.nim" \
  -exec cp /Users/martin/repos/nuwa/nuwa-sdk/src/nuwa_sdk.nim {} \;

# Build and test
nuwa develop
pytest ../nuwa-sdk/tests/test_numpy_integrations.py
```

### Compile Check Only

To verify the code compiles without running:

```bash
cd /Users/martin/repos/nuwa/nuwa-sdk
nim c --path:src tests/test_numpy.nim
```

## Common Issues and Solutions

### Issue: "Undeclared identifier: yourNewFunction"

**Cause:** The cached `nuwa_sdk.nim` wasn't updated.

**Solution:**
```bash
# Force refresh the cache
cd /Users/martin/repos/nuwa/nuwa-example/example_project
find .nimble/pkgs2 -name "nuwa_sdk.nim" \
  -exec cp /Users/martin/repos/nuwa/nuwa-sdk/src/nuwa_sdk.nim {} \;

# Clean and rebuild
nuwa clean
nuwa develop
```

### Issue: Tests Pass but Changes Don't Appear

**Cause:** The build system uses cached `.so` files.

**Solution:**
```bash
cd /Users/martin/repos/nuwa/nuwa-example/example_project
rm -rf example_project/*.so
nuwa develop
```

### Issue: Template/Generic Not Found

**Cause:** Templates need to be included in the same file for expansion.

**Solution:**
- Use `include nuwa_sdk` instead of `import nuwa_sdk` in Nim test files
- For actual usage, templates are properly expanded when called from exported functions

## Best Practices

### 1. Test Incrementally

After each significant change:
```bash
./tests/run_tests.sh
```

### 2. Keep Tests Focused

Test the specific feature you're working on:
```python
# Test only array summation
pytest tests/test_example_project.py::TestNumPyOperations::test_numpy_array_sum -v
```

### 3. Use Verbose Output for Debugging

```bash
pytest -v -s  # Show print statements
pytest --log-cli-level=DEBUG  # Show debug output
```

### 4. Test Edge Cases

Make sure to test:
- Empty arrays
- Single element arrays
- Large arrays (stress test)
- Different dtypes (int32, int64, float32, float64)
- Multi-dimensional arrays
- Error conditions

## Release Workflow

When ready to publish a new version:

```bash
# 1. Update version in nuwa_sdk.nimble
version = "0.4.0"

# 2. Commit changes
git add .
git commit -m "Release v0.4.0: Add numpy array wrappers"

# 3. Create and push tag
git tag v0.4.0
git push origin main
git push origin v0.4.0

# 4. Update dependent projects
# In nuwa-example/pyproject.toml:
nimble-deps = ["nimpy@0.2.1", "nuwa_sdk@0.4.0"]
```

## Quick Reference Commands

```bash
# === Test nuwa_sdk changes ===
./tests/run_tests.sh                          # Full test suite

# === Manual testing ===
find .nimble/pkgs2 -name "nuwa_sdk.nim" \
  -exec cp /path/to/nuwa-sdk/src/nuwa_sdk.nim {} \;  # Update cache
nuwa develop                               # Build
pytest                                       # Run tests

# === Troubleshooting ===
nuwa clean --all                             # Deep clean
rm -rf .nimble/pkgs2/nuwa_sdk-*            # Remove cached nuwa_sdk

# === Compile check only ===
nim c --path:src tests/test_numpy.nim     # Syntax check
```

## Project Structure Reference

```
nuwa/
├── nuwa-sdk/              # SDK library
│   ├── src/
│   │   └── nuwa_sdk.nim   # Main SDK (withNogil, nuwa_export, numpy)
│   ├── tests/
│   │   ├── test_numpy.nim           # Nim unit tests (compile check only)
│   │   ├── test_numpy_integrations.py  # Python integration tests
│   │   └── run_tests.sh            # Automated test runner
│   └── nuwa_sdk.nimble
│
├── nuwa-example/          # Example consuming project
│   └── example_project/
│       ├── pyproject.toml
│       ├── nim/
│       │   └── example_project_lib.nim
│       └── .nimble/             # Nimble cache (local nuwa_sdk copied here)
│
└── nuwa-build/            # Build system
```

## Testing by Feature

### Testing withNogil Changes

```bash
# 1. Modify src/nuwa_sdk.nim
# 2. Run tests
./tests/run_tests.sh

# 3. Or test GIL release specifically
python3 -c "
from example_project import numpy_array_sum_fast
import numpy as np
arr = np.arange(10000, dtype=np.int64)
result = numpy_array_sum_fast(arr)
print(f'Sum of 0..9999: {result}')
assert result == 49995000
"
```

### Testing numpy Wrapper Changes

```bash
# After modifying numpy wrappers in nuwa_sdk.nim:
./tests/run_tests.sh

# Test specific functionality
python3 -c "
import numpy as np
from example_project import numpy_matrix_multiply

mat_a = np.array([[1, 2], [3, 4]], dtype=np.float64)
mat_b = np.array([[5, 6], [7, 8]], dtype=np.float64)
result = numpy_matrix_multiply(mat_a, mat_b)
print(f'Matrix product:\\n{result}')
assert result == [[19.0, 22.0], [43.0, 50.0]]
"
```

### Testing nuwa_export Changes

```bash
# After modifying export macro:
./tests/run_tests.sh

# Verify stubs are generated
ls -la /Users/martin/repos/nuwa/nuwa-example/example_project/example_project/*.pyi
```

## Additional Resources

- [Nuwa README](README.md) - Main project documentation
- [Example Project](../nuwa-example/README.md) - Usage examples
- [Nimble Documentation](https://nimble.directory/docs/) - Package manager docs
- [Nim by Example](https://nim-by-example.github.io/) - Nim language reference
