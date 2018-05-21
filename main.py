def some_magic():
    import basicCV as a
    for i in dir(a):
        item = getattr(a,i)
        if callable(item):
            item()

if __name__ == '__main__':
    some_magic()