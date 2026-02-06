# Core Module Rules

## Role

Define system-wide shared types, interfaces, exceptions, and constants.
Other modules depend on this module, but this module has ZERO dependencies on other `src/` modules.

## Rules

- When adding new fields to types.py dataclasses, always provide default values (backward compatibility)
- Enum value changes forbidden — only additions allowed
- interfaces.py ABC signature changes require simultaneous updates to ALL implementations
- All magic numbers/strings go in constants.py — no literals in other files
- All dataclasses must have `__post_init__` validation where applicable
- Use `uuid7` for all ID generation (time-ordered)
