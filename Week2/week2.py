#exc1
import numpy as np
"""
arr = np.arange(12,38,1)
reverse_arr = np.flip(arr) #[::-1]
print(reverse_arr)
"""
#exc2
"""
Array1 = np.array([ 0, 10, 20, 40, 60])
Array2 = np.array([10, 30, 40])
result = []
for element in Array1:
    if element in Array2:
        result.append(True)
    else:
        result.append(False)
print("result: ",result)
"""
#exc3
"""
Arr=np.array([1,6,4,8,9,-4,-2,11])
max=min=Arr[0]
for i in Arr:
    if i > max:
        max = i
    elif i < min:
        min = i
print("Max element:",max)
print("Min element:",min)
"""
#exc4
import string
from collections import Counter

file_patch="D:\AI\VietAI-x-CoderSchool\Week2\story.txt"
with open(file_patch, "r") as file:
    text = file.read()
text = text.lower()  
text = text.translate(str.maketrans("", "", string.punctuation))  
words = text.split()
word_counts = Counter(words)
most_common_words = word_counts.most_common(100)
for word, count in most_common_words:
    print(f"{word}: {count}")   


