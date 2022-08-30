import datetime
a = datetime.datetime.now()
for i in range(5000):
    for i in range(5000):
        s = i * 2
b = datetime.datetime.now() - a
print(b)
print(b.seconds, b.microseconds)