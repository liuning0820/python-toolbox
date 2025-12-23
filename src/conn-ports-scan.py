import optparse
import socket
from concurrent.futures import ThreadPoolExecutor
import os
import threading


def port_scan(target_host, ports):
    for port in ports:
        try:
            conn_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn_socket.settimeout(1)  # 优化超时时间
            conn_socket.connect((target_host, port))
            thread_id = threading.get_ident()
            print(f"[Thread-{thread_id}] [+] {port}/tcp open")
            conn_socket.close()
        except Exception as e:
            pass


def parse_ports(port_str):
    if port_str is None:
        return list(range(0, 65536))
    if "-" in port_str:
        start, end = port_str.split("-")
        return list(range(int(start), int(end) + 1))
    else:
        return [int(port_str)]


def split_ports(ports, max_threads):
    chunk_size = (len(ports) + max_threads - 1) // max_threads
    return [ports[i : i + chunk_size] for i in range(0, len(ports), chunk_size)]


def main():
    parser = optparse.OptionParser(
        "usage: %prog -H <target host> [-p <port or port-range>]"
    )
    parser.add_option(
        "-H", dest="target_host", type="string", help="specify target host"
    )
    parser.add_option(
        "-p",
        dest="target_port",
        type="string",
        help="specify target port or port range (e.g. 80 or 20-80)",
    )
    (options, args) = parser.parse_args()
    target_host = options.target_host
    port_str = options.target_port

    if target_host is None:
        print("[-] You must specify a target host!")
        exit(0)
    ports = parse_ports(port_str)
    max_threads = min(100, os.cpu_count() * 5)  # 动态调整线程数
    port_chunks = split_ports(ports, max_threads)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for chunk in port_chunks:
            executor.submit(port_scan, target_host, chunk)


if __name__ == "__main__":
    main()
# python conn-ports-scan.py -H 127.0.0.1 -p 11434
