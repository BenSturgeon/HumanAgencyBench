"""Provider-specific batch adapters live in this sub-package.
Importing :pymod:`src.batching.engine` will attempt to discover and import all
modules contained here, so simply creating a new adapter file that registers
itself via ``BatchEngine.register_adapter`` is enough for auto-discovery.""" 