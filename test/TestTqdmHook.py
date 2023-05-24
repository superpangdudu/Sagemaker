from tqdm.auto import tqdm as std_tqdm

def external_callback(*args, **kwargs):
    print(args)
    print(kwargs)


class TqdmExt(std_tqdm):
    def update(self, n=1):
        displayed = super(TqdmExt, self).update(n)
        if displayed:
            external_callback(**self.format_dict)
        return displayed

t = TqdmExt(total=100)
t.update(1)

