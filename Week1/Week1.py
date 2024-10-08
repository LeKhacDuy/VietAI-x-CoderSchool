#exc1
"""
def number(data1):
    positive = 0
    negative = 0
    for i in data1:
        if i > 0:
            positive += 1
        else: 
            negative += 1
    return positive,negative
data1 = [-10, -21, -4, -45, -66, 93, 11, -4, -6, 12, 11, 4]
positive,negative = number(data1)
print("positive = ",positive)
print("negative = ",negative)
"""
#exc2
"""
def number(data2):
    k = 3
    element = 0
    for i in data2:
        if i > k:
            element += 1
    return element
data2 = [4, 6, 4, 3, 3, 4, 3, 4, 3, 8]
element = number(data2)
print("element: ",element)
"""
#exc3
"""
def number(data3):
    n = []
    for i in range (1, len(data3)):
        if data3[i] > data3[i-1]:
            n.append(data3[i])
    return n
data3 = [4, 5, 6, 7, 3, 9, 11, 2, 10]
n = number(data3)
print(n)
"""
#exc4
"""
def combienumber(data4):
    combination_numbers =[]
    for i in range (len(data4)):
        for j in range (len(data4)):
            if i!=j:
                for k in range(len(data4)):
                    if k!=j and k!=i:
                        combination_number = int(f"{data4[i]}{data4[j]}{data4[k]}")
                        combination_numbers.append(combination_number)
    return combination_numbers
data4 = [1,2,3]
combination_numbers = combienumber(data4)
print(combination_numbers)
"""
#exc5
"""
data5_list1 = [[4, 3, 5, ], [1, 2, 3], [3, 7, 4]]
data5_list2 = [[1, 3], [9, 3, 5, 7], [8]]
for i in range (len(data5_list1)):
    data5_list1[i].extend(data5_list2[i])
   
print(data5_list1)
"""
#exc6
"""
numbers = []
for i in range (2000,3201):
    if i % 7 == 0 and i % 5 !=0:
        numbers.append(str(i))
print(",".join(numbers))
"""
#exc7
"""
numbers = []
for i in range(1000,3001):
    if i % 7 == 0 and i % 5 !=0 and i % 2 == 0:
         numbers.append(str(i))
print(",".join(numbers))
"""