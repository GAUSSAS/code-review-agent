"""支持 `python -m code_review_agent ...` 方式调用。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
