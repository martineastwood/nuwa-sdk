# nuwa_sdk

**Compile-time metadata generation for Nuwa Build**

This package provides the `nuwa_export` macro that enables automatic generation of Python type stubs (`.pyi` files) for Nim extensions built with Nuwa Build.

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

## How It Works

The `nuwa_export` macro:

- Inspects your Nim functions at compile time
- Extracts parameter names, types, return types, and docstrings
- Maps Nim types to Python type annotations
- Outputs JSON metadata to the compiler's stdout
- The `nuwa_build` Python tool captures this output and generates `.pyi` files

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
