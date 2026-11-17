# Writing an adapter

1. Implement `Driver` (`openenv_daw.adapters`): `read()` (project without notes; object ids are the
   DAW's own stable keys), `read_notes(clip)`, `apply(changes, ids)` (atomic, raise `DriverError` on failure; bind every created
   object with `ids.bind(canonical_id, native_key)`), `close()`.
2. Describe devices in a `DeviceMap` YAML (`native` name, per-parameter `native`, optional
   `min`/`max`, or `unsupported`).
3. Wrap it: `MirrorBackend(driver, device_map)`.
4. Run `assert_driver_conformance(make_driver, tasks, device_map=...)` over the held-out tasks and a few
   train samples; `FakeDriver` in `openenv_daw.adapters.testing` is a complete worked example.
5. Register a factory `make_backend()` under the `openenv_daw.backends` entry point group;
   serve with `openenv-daw mcp --backend NAME`.

Destructive loading (replacing the whole project, needed for benchmarks) is refused unless the
project contains a track named `__openenv_scratch__` and you pass `allow_destructive_load=True`.
