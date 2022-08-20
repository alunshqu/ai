class Demo:
    # c1 = 50
    # c2 = 20
    def __int__(self):
        self.c1 = 10
        self.c2 = 20

demo = Demo()
demo.__int__()
# demo.c1 = 10
print(demo.c1)