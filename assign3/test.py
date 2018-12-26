import os

print(os.path.realpath(__file__))
print(os.getcwd())
print("test1.py")
print(os.path.normpath("test2.py"))
print(os.path.relpath("test3.py"))
print(os.path.realpath("test4.py"))

