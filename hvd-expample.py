
def fn(magic_number):
    import horovod.torch as hvd
    hvd.init()
    print(f'Hey you Ahole, this is me,' +\
            f' rank = {hvd.rank()},' +\
            f' local_rank = {hvd.local_rank()},' +\
            f' size = {hvd.size()},' +\
            f' local_size = {hvd.local_size()},' +\
            f' magic_number = {magic_number}')
    return hvd.rank()

import horovod.spark
horovod.spark.run(fn, args=(42,))