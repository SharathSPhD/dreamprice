# Python Style Guide — DreamPrice

## Tooling
- **Formatter:** `ruff format` (line length 100)
- **Linter:** `ruff check` with C901 (max McCabe complexity 10)
- **Cognitive complexity:** `complexipy` (max 15)
- **Type checker:** `pyright` in strict mode

## Type Annotations
- All public functions must have full type annotations
- Use `torch.Tensor` not `Any` for tensor arguments
- Use `nn.Module` subclasses with typed `forward()` signatures
- Prefer `TypedDict` over bare `dict` for structured data

## Naming
- Tensors: name by shape suffix, e.g. `x_BTC` means `(Batch, Time, Channels)`
- Math variables match blueprint notation: `z_t`, `h_t`, `beta_pred`, `lambda_lcb`
- Constants in `UPPER_SNAKE_CASE`, model params in `lower_snake_case`

## Module Structure
Each module file must have:
1. Module-level docstring citing the blueprint section it implements
2. `__all__` list of public names
3. No circular imports — dependency order: `utils` → `data` → `models` → `training` → `inference` → `api`

## PyTorch Conventions
- Use `nn.Module` with `forward()` — never bare functions for stateful components
- Device management via `utils/device.py` — never hardcode `.cuda()` or `.cpu()`
- Always call `model.train()` / `model.eval()` explicitly before use
- BF16: cast inputs at boundary, keep SSM state in FP32 explicitly

## Error Handling
- Raise typed exceptions (e.g., `WeakInstrumentError`, `TemporalLeakageError`)
- Validate at data boundaries (loader, API endpoints) — trust internal tensor shapes
- Never silently catch exceptions in training loop

## Comments
- Cite blueprint: `# Blueprint §4: DRAMA-style decoupled posterior`
- Cite paper: `# DreamerV3 App. C: KL balancing, beta_dyn=0.5, beta_rep=0.1`
- No inline comments for obvious operations
