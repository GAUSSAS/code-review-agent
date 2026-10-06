"""文件夹选择对话框（独立进程运行）。

为什么要单独一个进程？
----------------------
1. **tkinter 必须在主线程创建窗口**，而 HTTP 请求处理发生在工作线程里；
   直接在 handler 中调用 ``askdirectory()`` 会抛出
   ``RuntimeError: main thread is not in main loop``。
2. 独立进程还带来两个好处：对话框阻塞（用户可能停留很久）不会占用
   服务器的线程池；用户取消或直接关掉窗口也不会影响服务。

协议（供 server.py 调用）：**把选中的绝对路径打印到 stdout 并退出**。
未选择时打印空行并以退出码 0 结束——「取消」不是错误。
真正出错（例如没有图形环境）走 stderr + 非 0 退出码。
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="弹出系统文件夹选择对话框，输出选中路径")
    parser.add_argument("--initial", default="", help="初始目录")
    parser.add_argument("--title", default="选择被审查目录", help="对话框标题")
    args = parser.parse_args(argv)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        print(
            f"缺少 tkinter，无法弹出图形化目录选择框：{exc}。"
            "请改为手动填写路径。",
            file=sys.stderr,
        )
        return 2

    try:
        root = tk.Tk()
    except Exception as exc:  # noqa: BLE001 - 无图形环境（服务/远程会话）会失败
        print(
            f"无法初始化图形界面（可能没有桌面环境）：{type(exc).__name__}: {exc}。"
            "请改为手动填写路径。",
            file=sys.stderr,
        )
        return 3

    # 隐藏主窗口，只保留对话框
    root.withdraw()
    root.title(args.title)
    try:
        root.attributes("-topmost", True)
    except Exception:  # noqa: BLE001 - 部分平台不支持该属性
        pass

    try:
        selected = filedialog.askdirectory(
            title=args.title,
            initialdir=args.initial or None,
            mustexist=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"打开目录选择框失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass

    # 取消时 selected 为空字符串，打印空行表示「没有选择」（不是错误）
    print(selected or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
