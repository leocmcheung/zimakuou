def test_modules_import_cleanly():
    """Catches syntax errors and circular imports without loading any models."""
    import zimakuou
    import zimakuou.audio  # noqa: F401
    import zimakuou.context  # noqa: F401
    import zimakuou.pipeline  # noqa: F401
    import zimakuou.srt_writer  # noqa: F401
    import zimakuou.transcribe  # noqa: F401
    import zimakuou.translate  # noqa: F401
    import zimakuou.translators  # noqa: F401
    import zimakuou.translators.base  # noqa: F401
    from zimakuou.translators._prompt import SYSTEM, build_prompt, build_system  # noqa: F401

    assert zimakuou.__version__
