import torchvision
from torchvision import datasets, transforms, utils
import torch.utils.data
from torch.utils.data import Dataset, DataLoader


# Define the transform option
transform_list = [
                  transforms.Resize((256, 128), interpolation=3),
                  transforms.Pad(10),
                  transforms.RandomCrop((256, 128)),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ]

# Send the option to the transformer, for example: for train, for val, for test
data_transformer = {
    'train': transforms.Compose(transform_list),
    'val': transforms.Compose(transform_list),
    'test': transforms.Compose(transform_list),
}


# Define the Data Loader
# torchvision.datasets.ImageFolder  is for easier and format read the data
#
# the image data should be stored in one big folder contains each class, a class map a file folder, and all original data are in the class folder
#   ./data_root
#       /class_name(always neumerical)
#                   /data(.jpg,.png etc)
#
# the default loader will be PIL image
# target_transform is for transforming the label
image_sets['train'] = datasets.ImageFolder(data_root, data_transformer['train'], target_transform=None, loader=default_loader)
image_sets['train'] = datasets.ImageFolder(data_root, data_transformer['val'], target_transform=None, loader=default_loader)
image_sets['train'] = datasets.ImageFolder(data_root, data_transformer['test'], target_transform=None, loader=default_loader)


# Read and transform the data
# it is freedom by using torch.utils.data.DataLoader, you can be the King of your data, control them to come in
data_loaders = { x : torch.utils.data.DataLoader(image_sets[x], batch_size, shuffle = False, num_workers = 8) for x in ['train', 'val', 'test']}

# Let's see what we got now
#
#
#
# dataloaders[] is a iterable object and it contains the input data and classes of each data
# also it contains all the data and iteral total / batch_size (almost) time
# so if we use
print(len(next(iter(dataloaders['train'])))) # output : 2 (input and class)
# and we can see it matched
print(next(iter(dataloaders['train']))[0].shape) # output : batch_size
print(next(iter(dataloaders['train']))[1].shape) # output : batch_size
# if we choose one of the data
print(next(iter(dataloaders['train']))[0][0].shape) # output : torch.Size([3, 256, 128])
print(next(iter(dataloaders['train']))[1][0]) # output : label

# We load the data by ourselves, thanks for helping: https://blog.csdn.net/tfygg/article/details/73354235
# It is very important to understand the torch.utils.data.Dataset and torch.utils.data.DataLoader (Revision the python version at the bottom)
#
#
# Torch.utils.data.Dataset
class Dataset(object):
    """An abstract class representing a Dataset.
        
        All other datasets should subclass it. All subclasses should override
        ``__len__``, that provides the size of the dataset, and ``__getitem__``,
        supporting integer indexing in range from 0 to len(self) exclusive.
        """
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __add__(self, other):
        return ConcatDataset([self, other])


# Torch.utils.data.DataLoader
class DataLoader(object):
    r"""
        Data loader. Combines a dataset and a sampler, and provides
        single- or multi-process iterators over the dataset.
        
        Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
        (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
        at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
        the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
        indices at a time. Mutually exclusive with batch_size, shuffle,
        sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
        loading. 0 means that the data will be loaded in the main process.
        (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
        into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
        from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
        worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
        input, after seeding and before data loading. (default: None)
        
        .. note:: By default, each worker will have its PyTorch seed set to
        ``base_seed + worker_id``, where ``base_seed`` is a long generated
        by main process using its RNG. However, seeds for other libraies
        may be duplicated upon initializing workers (w.g., NumPy), causing
        each worker to return identical random numbers. (See
        :ref:`dataloader-workers-random-seed` section in FAQ.) You may
        use ``torch.initial_seed()`` to access the PyTorch seed for each
        worker in :attr:`worker_init_fn`, and use it to set other seeds
        before data loading.
        
        .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
        unpicklable object, e.g., a lambda function.
        """
    
    __initialized = False
    
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        
        if timeout < 0:
            raise ValueError('timeout option should be non-negative')
        
        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = None
        
        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')
        
        if self.num_workers < 0:
            raise ValueError('num_workers option cannot be negative; '
                             'use num_workers=0 to disable multiprocessing.')
        
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True
    
    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))
        
        super(DataLoader, self).__setattr__(attr, val)
    
    def __iter__(self):
        return _DataLoaderIter(self)
    
    def __len__(self):
        return len(self.batch_sampler)

class _DataLoaderIter(object):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""
    
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.done_event = threading.Event()
        
        self.sample_iter = iter(self.batch_sampler)
        
        base_seed = torch.LongTensor(1).random_().item()
        
        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queues = [multiprocessing.Queue() for _ in range(self.num_workers)]
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}
            
            self.workers = [
                            multiprocessing.Process(
                                                    target=_worker_loop,
                                                    args=(self.dataset, self.index_queues[i],
                                                          self.worker_result_queue, self.collate_fn, base_seed + i,
                                                          self.worker_init_fn, i))
                            for i in range(self.num_workers)]
                
                            if self.pin_memory or self.timeout > 0:
                                self.data_queue = queue.Queue()
                                if self.pin_memory:
                                    maybe_device_id = torch.cuda.current_device()
                                        else:
                                            # do not initialize cuda context if not necessary
                                            maybe_device_id = None
                                                self.worker_manager_thread = threading.Thread(
                                                                                              target=_worker_manager_loop,
                                                                                              args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
                                                                                                    maybe_device_id))
                                                                                                  self.worker_manager_thread.daemon = True
                                                                                                      self.worker_manager_thread.start()
                                                                                                  else:
                                                                                                      self.data_queue = self.worker_result_queue
                                                                                                          
                                                                                                          for w in self.workers:
                                                                                                              w.daemon = True  # ensure that the worker exits on process exit
                                                                                                                  w.start()
                                                                                                                      
                                                                                                                      _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
                                                                                                                      _set_SIGCHLD_handler()
                                                                                                                      self.worker_pids_set = True
                                                                                                                          
                                                                                                                          # prime the prefetch loop
                                                                                                                          for _ in range(2 * self.num_workers):
                                                                                                                              self._put_indices()

def __len__(self):
    return len(self.batch_sampler)
    
    def _get_batch(self):
        if self.timeout > 0:
            try:
                return self.data_queue.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self.timeout))
        else:
            return self.data_queue.get()

def __next__(self):
    if self.num_workers == 0:  # same-process loading
        indices = next(self.sample_iter)  # may raise StopIteration
        batch = self.collate_fn([self.dataset[i] for i in indices])
        if self.pin_memory:
            batch = pin_memory_batch(batch)
            return batch
        
        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)
        
        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration
    
    while True:
        assert (not self.shutdown and self.batches_outstanding > 0)
        idx, batch = self._get_batch()
        self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
        return self._process_next_batch(batch)

next = __next__  # Python 2 compatibility

def __iter__(self):
    return self
    
    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1
    
    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch
    
    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("_DataLoaderIter cannot be pickled")
    
    def _shutdown_workers(self):
        try:
            if not self.shutdown:
                self.shutdown = True
                self.done_event.set()
                for q in self.index_queues:
                    q.put(None)
                # if some workers are waiting to put, make place for them
                try:
                    while not self.worker_result_queue.empty():
                        self.worker_result_queue.get()
                except (FileNotFoundError, ImportError):
                    # Many weird errors can happen here due to Python
                    # shutting down. These are more like obscure Python bugs.
                    # FileNotFoundError can happen when we rebuild the fd
                    # fetched from the queue but the socket is already closed
                    # from the worker side.
                    # ImportError can happen when the unpickler loads the
                    # resource from `get`.
                    pass
                # done_event should be sufficient to exit worker_manager_thread,
                # but be safe here and put another None
                self.worker_result_queue.put(None)
        finally:
            # removes pids no matter what
            if self.worker_pids_set:
                _remove_worker_pids(id(self))
                self.worker_pids_set = False

def __del__(self):
    if self.num_workers > 0:
        self._shutdown_workers()

'''
In python:

__len__()
__getitem__()


and

__iter__()
__next__()


class CountList:
    def __init__(self, *args):
        self.values = [x for x in args]
        self.count = {}.fromkeys(range(len(self.values)),0)
    # 这里使用列表的下标作为字典的键，注意不能用元素作为字典的键
    # 因为列表的不同下标可能有值一样的元素，但字典不能有两个相同的键
    def __len__(self):
        return len(self.values)
    def __getitem__(self, key):
        self.count[key] += 1
        return self.values[key]
c1 = CountList(1,3,5,7,9)
c2 = CountLIst(2,4,6,8,10)

# 调用
c1[1]  ## 3
c2[1]  ## 4
c1[1] + c2[1]     ## 7
c1.count  ## {0:0,1:2,2:0,3:0,4:0}
c2.count  ## {0:0,1:2,2:0,3:0,4:0}


class Fibs:
    def __init__(self, n=20):
        self.a = 0
        self.b = 1
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > self.n:
            raise StopIteration
        return self.a

## 调用
fibs = Fibs()
for each in fibs:
    print(each)
## 输出
1
1
2
3
5
8
13

'''
# Thanks: https://blog.csdn.net/qq_36653505/article/details/83351808 and https://blog.csdn.net/u014380165/article/details/79058479
