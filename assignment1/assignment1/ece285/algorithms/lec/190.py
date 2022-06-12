def reverseBits(n: int) -> int:
    res = n & 1
    i = 0
    while(i<32):
        i = i + 1
        n = n >> 1
        res = (res << 1) + (n & 1)


    return res
print(bin(reverseBits(43261596)))
