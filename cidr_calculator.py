import ipaddress

def calculate_cidr_and_iprange(ip_address, mask_bits):
    # 创建IPv4Network对象
    network = ipaddress.IPv4Network(f"{ip_address}/{mask_bits}", strict=False)
    
    # 获取CIDR表示法
    cidr_notation = str(network)
    
    # 计算网络地址和广播地址
    network_address = network.network_address
    broadcast_address = network.broadcast_address
    
    # 计算IP范围
    ip_range_start = network_address + 1
    ip_range_end = broadcast_address - 1
    
    # 返回结果
    return {
        "CIDR Notation": cidr_notation,
        "Network Address": str(network_address),
        "Broadcast Address": str(broadcast_address),
        "IP Range Start": str(ip_range_start),
        "IP Range End": str(ip_range_end)
    }

# 示例：计算IP地址 "172.18.190.57" 和 子网掩码位数 20 的CIDR表示法及IP范围
result = calculate_cidr_and_iprange("172.18.190.57", 20)
print(result)

def main():
    # 用户输入IP地址
    ip_address = input("请输入IP地址(例如:192.168.1.0): ")
    
    # 用户输入子网掩码位数
    mask_bits = int(input("请输入子网掩码位数(例如:24): "))
    
    # 计算并打印结果
    result = calculate_cidr_and_iprange(ip_address, mask_bits)
    print("\n计算结果:")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()