#!/usr/bin/env python3
"""极简 HTTP CONNECT 代理（纯直连出口，不做上游转发）。

背景：本机 dockerd 被 /etc/systemd/system/docker.service.d/http-proxy.conf 强制
要求走 127.0.0.1:7890，而该端口没有服务，导致 `docker pull` 一律报
    proxyconnect tcp: dial tcp 127.0.0.1:7890: connect: connection refused
该文件属 root，不去动它；改在 7890 上放一个只做直连隧道的代理，dockerd 的
要求即被满足，`docker pull` 恢复正常（已实测：既有的 isaac-lab:2.3.2 报
"Image is up to date"，全新 busybox 真下载成功）。

只监听 127.0.0.1，只支持 CONNECT（docker 拉镜像只需要它），逐连接开线程转发，
不落盘任何流量日志。用 `env.sh proxy start|stop|status` 启停。
"""
import selectors
import socket
import sys
import threading

LISTEN = ("127.0.0.1", 7890)
BUF = 65536


def relay(a: socket.socket, b: socket.socket) -> None:
    sel = selectors.DefaultSelector()
    sel.register(a, selectors.EVENT_READ)
    sel.register(b, selectors.EVENT_READ)
    try:
        while True:
            for key, _ in sel.select(timeout=300):
                other = b if key.fileobj is a else a
                data = key.fileobj.recv(BUF)
                if not data:
                    return
                other.sendall(data)
    except OSError:
        return
    finally:
        sel.close()


def read_headers(conn: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 65536:
        chunk = conn.recv(BUF)
        if not chunk:
            break
        data += chunk
    return data


def handle(conn: socket.socket, addr) -> None:
    try:
        head = read_headers(conn)
        if not head:
            return
        parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            conn.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            return
        host, _, port = parts[1].partition(":")
        upstream = socket.create_connection((host, int(port or 443)), timeout=30)
        conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        relay(conn, upstream)
    except Exception as e:  # noqa: BLE001
        print(f"[proxy] {addr} 失败: {e}", flush=True)
    finally:
        try:
            conn.close()
        except OSError:
            pass


def main() -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(LISTEN)
    srv.listen(64)
    print(f"[proxy] 监听 {LISTEN[0]}:{LISTEN[1]} → 直连出口", flush=True)
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
