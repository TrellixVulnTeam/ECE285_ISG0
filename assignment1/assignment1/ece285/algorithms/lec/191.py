def hammingWeight(n: int) -> int:
    # assert len(str(n)) == 32
    string = str(bin(n))
    count = 0
    for i in string:
        if i=='1':
            count+=1
    return count

w = hammingWeight( 1111000 )
print(w)

