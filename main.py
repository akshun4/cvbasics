def run_all():
    import basicCV as a
    for i in dir(a):
        item = getattr(a,i)
        if callable(item):
            item()

if __name__ == '__main__':
    run_all()
