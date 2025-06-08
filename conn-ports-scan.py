import optparse
import socket

def port_scan(targetHost, port):
    try:
        connSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connSocket.settimeout(2)
        connSocket.connect((targetHost, port))
        print(f'[+] {port}/tcp open')
        connSocket.close()
    except Exception:
        print(f'[-] {port}/tcp closed')

def parse_ports(port_str):
    if port_str is None:
        return range(0, 65536)
    if '-' in port_str:
        start, end = port_str.split('-')
        return range(int(start), int(end) + 1)
    else:
        return [int(port_str)]

def main():
    parser = optparse.OptionParser('usage: %prog -H <target host> [-p <port or port-range>]')
    parser.add_option('-H', dest='targetHost', type='string', help='specify target host')
    parser.add_option('-p', dest='targetPort', type='string', help='specify target port or port range (e.g. 80 or 20-80)')
    (options, args) = parser.parse_args()
    targetHost = options.targetHost
    port_str = options.targetPort

    if targetHost is None:
        print('[-] You must specify a target host!')
        exit(0)
    ports = parse_ports(port_str)
    for port in ports:
        port_scan(targetHost, port)

if __name__ == '__main__':
    main()

# python conn-ports-scan.py -H 127.0.0.1 -p 11434