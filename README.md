# nuwa-sdk

**Python interoperability utilities for [Nuwa Build](https://github.com/martineastwood/nuwa-build)**

This package provides utilities for building high-performance Python extensions with Nim:

- **`nuwa_export`** - Macro for automatic generation of Python type stubs (`.pyi` files)
- **`withNogil`** - Template for releasing the Python GIL during pure Nim code execution

## Usage

```nim
import nuwa_sdk

proc add(a: int, b: int): int {.nuwa_export.} =
  ## Add two integers together
  return a + b

proc greet(name: string): string {.nuwa_export.} =
  ## Greet a person by name
  return "Hello, " & name
```

When you build your project with `nuwa build`, this macro will:

1. Export the function to Python (via `nimpy`)
2. Emit compile-time metadata about function signatures
3. Enable automatic generation of `.pyi` stub files

## GIL Release with `withNogil`

The `withNogil` template allows you to release Python's Global Interpreter Lock (GIL) during performance-critical Nim code. This enables true parallelism when your extension is used from multi-threaded Python code.

```nim
import nuwa_sdk

proc computePi(iterations: int): float {.nuwa_export.} =
  ## Compute Pi using Monte Carlo method (GIL-released)
  var count = 0
  withNogil:
    # Pure Nim code runs without Python interpreter interference
    # Other Python threads can run concurrently
    for i in 0..<iterations:
      let x = rand(1.0)
      let y = rand(1.0)
      if x*x + y*y <= 1.0:
        count += 1
  return 4.0 * float(count) / float(iterations)
```

### When to use `withNogil`

- **CPU-bound computations**: Mathematical operations, numeric algorithms
- **Batch processing**: Processing large datasets without Python interaction
- **Parallel workloads**: Allow other Python threads to run while Nim computes

### When NOT to use `withNogil`

- **Python API calls**: Any code that calls Python objects or nimpy functions
- **Data transfer**: Converting between Nim and Python types
- **I/O operations**: File/network operations that release GIL automatically

### How it works

1. `PyEval_SaveThread()` releases the GIL
2. Pure Nim code executes without Python interference
3. `PyEval_RestoreThread()` reacquires the GIL before returning to Python

Equivalent to Cython's `with nogil:` block or CPython's `Py_BEGIN_ALLOW_THREADS`.

## How It Works

### `nuwa_export` macro

- Inspects your Nim functions at compile time
- Extracts parameter names, types, return types, and docstrings
- Maps Nim types to Python type annotations
- Outputs JSON metadata to the compiler's stdout
- The `nuwa_build` Python tool captures this output and generates `.pyi` files

### `withNogil` template

The `withNogil` template:
- Wraps a block of Nim code with GIL release/reacquire calls
- Dynamically loads Python C API functions (lazy initialization)
- Ensures GIL is always reacquired even if an exception occurs
- Allows multiple Python threads to execute Nim code concurrently

## Supported Type Mappings

| Nim Type                      | Python Type |
| ----------------------------- | ----------- |
| `int`, `int32`, `int64`       | `int`       |
| `float`, `float32`, `float64` | `float`     |
| `string`                      | `str`       |
| `bool`                        | `bool`      |
| `void`                        | `None`      |
| `seq[T]`                      | `list[T]`   |
| `array[N, T]`                 | `list[T]`   |
| Other types                   | `Any`       |

## Installation

This package is automatically installed as a dependency when you use `nuwa new` to create a new project.

You can also install it manually:

```bash
nimble install nuwa_sdk
```

## License

MIT
