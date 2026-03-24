class MnemoError(Exception):
    pass


class StoreError(MnemoError):
    pass


class EmbeddingError(MnemoError):
    pass


class NamespaceError(MnemoError):
    pass
