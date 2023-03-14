import os
import time
import warnings
import multiprocessing as mp
import numpy as np

# import logging
# mp.log_to_stderr()
# logger = mp.get_logger()
# logger.setLevel(logging.INFO)

class Config(object):
    """ A Config Interface. """
    def __init__(self, **kwargs):
        """Set values of computed attributes."""
        for k in kwargs:
            if k in dir(self):
                setattr(self, k, kwargs[k])
    
    def __str__(self):
        res = []
        for x in dir(self):
            if not x.startswith("__") and not callable(getattr(self, x)):
                res.append("{} = {}".format(x, self._display_config(x)))
        return '\n'.join(res) + '\n'
    
    def display(self):
        """Display Configuration values."""
        for x in dir(self):
            if not x.startswith("__") and not callable(getattr(self, x)):
                print("{:30} {}".format(x, self._display_config(x)))
    
    def _display_config(self, x):
        return repr(getattr(self, x))


def config_cmd_parser(config):
    members = [attr for attr in dir(config) 
               if not callable(getattr(config, attr)) and not attr.startswith("__")]
    return members


def pc_link(inputs, outputs, processor=None, container=None, logging=False):
    """ A generic function to connect a producer and a consumer.
        Parallel the following process:
        in_queue -> processor -> out_queue
        
        inputs: (in_queue, n_in_process, counter, locker, node_id) or None
            counter is a Manager().Value() to record exited processes.
        outputs: (out_queue, n_out_process, counter, locker, node_id) or None
        processor: (fn, *args, **kwargs) or None, fn should be a iterator.
        container: if container is provided, function will store values to 
            an external Manager().list() for a copy.
        
        1). outputs != None, inputs != None, processor != None:
            pop x from in_queue (store in container) -> analyze x -> push result into out_queue
        2). outputs != None, inputs != None, processor == None:
            pop x from in_queue (store in container) -> push x into out_queue
        3). outputs != None, inputs == None, processor != None:
            run processor fn and yield result into out_queue (no container)
        4). outputs != None, inputs == None, processor == None:
            raise Error
        5). outputs == None, inputs != None, processor != None:
            pop x from in_queue (store in container) -> analyze it (save to file),
        6). outputs == None, inputs != None, processor == None:
            pop x from in_queue (store in container), most likely get all values 
            from the last queue and done. If container is not given, will prompt warning.
        7). outputs == None, inputs == None:
            raise Error
        
        Note: always provide a counter if inputs n_in_process > 1, 
        otherwise next chain may lose information. 
    """
    def _get_fn(processor):
        fn = processor.setdefault('fn', lambda x: x)
        args = processor.setdefault('args', ())
        kwargs = processor.setdefault('kwargs', {})
        return fn, args, kwargs
    
    # print([inputs, outputs, processor, container])
    # fn, args, kwargs = _get_fn(processor)
    # print([fn, args, kwargs])
    
    if outputs is not None:
        out_queue, n_out_process, _, _, out_node_id = outputs
        if inputs is not None:
            in_queue, n_in_process, counter, lock, in_node_id = inputs
            if processor is not None:
                fn, args, kwargs = _get_fn(processor)
                if container is not None:
                    for x in iter(in_queue.get, None):
                        if logging:
                            print('Worker %s process item: %s' % (mp.current_process(), str(x)))
                        container.append(x)
                        try:
                            for _ in fn(x, *args, **kwargs):
                                try:
                                    out_queue.put(_)
                                except:
                                    print("%s cannot be push into out_queue %s" % (str(_), out_queue_id))
                                    raise
                        except ValueError as err:
                            print("Function %s (%s, %s) failed to process elements %s on worker %s from in_queue %s to out_queue %s" % 
                                  (fn.__name__, str(args), str(kwargs), str(x), mp.current_process(), in_queue_id, out_queue_id))
                            print("Value error: {0}".format(err))
                            raise
                else:
                    for x in iter(in_queue.get, None):
                        if logging:
                            print('Worker %s process item: %s' % (mp.current_process(), str(x)))
                        try:
                            for _ in fn(x, *args, **kwargs):
                                try:
                                    out_queue.put(_)
                                except: 
                                    print("%s cannot be push into out_queue %s" % (str(_), out_queue_id))
                                    raise
                        except ValueError as err:
                            print("Function %s (%s, %s) failed to process elements %s on worker %s from in_queue %s to out_queue %s" % 
                                  (fn.__name__, str(args), str(kwargs), str(x), mp.current_process(), in_queue_id, out_queue_id))
                            print("Value error: {0}".format(err))
                            raise
            else:
                if container is not None:
                    for x in iter(in_queue.get, None):
                        if logging:
                            print('Worker %s process item: %s' % (mp.current_process(), str(x)))
                        container.append(x)
                        out_queue.put(x)
                else:
                    for x in iter(in_queue.get, None):
                        if logging:
                            print('Worker %s process item: %s' % (mp.current_process(), str(x)))
                        out_queue.put(x)
            
            lock.acquire()
            counter.value += 1
            if logging:
                print("Finished")
                print('Worker %s assign counter to: %s' % (mp.current_process(), counter.value))
            if counter.value < n_in_process:
                lock.release()
                return
            if logging:
                print("Last process fill None into out_queue")
            for _ in range(n_out_process):
                out_queue.put(None)
            lock.release()
            
        else:
            in_queue, n_in_process, counter, lock, in_node_id = None, 1, None, None, None
            if processor is not None:
                fn, args, kwargs = _get_fn(processor)
                try:
                    for x in fn(*args, **kwargs):
                        if logging:
                            print('Worker %s process item: %s' % (mp.current_process(), str(x)))
                        try:
                            out_queue.put(x)
                        except:
                            print("%s cannot be push into out_queue %s" % (str(x), out_queue_id))
                            raise
                except ValueError as err:
                    print("Function %s (%s, %s) failed to yield elements into out_queue %s" % 
                          (fn.__name__, str(args), str(kwargs), out_queue_id))
                    print("Value error: {0}".format(err))
                    raise
            else:
                raise ValueError("Inputs and processor cannot be both None.")
            
            if logging:
                print("Finished")
                print("Last process fill None into out_queue")
            for _ in range(n_out_process):
                out_queue.put(None)

    else:
        n_out_process = 0
        if inputs is not None:
            in_queue, n_in_process, counter, lock, in_node_id = inputs
            if processor is not None:
                fn, args, kwargs = _get_fn(processor)
                if container is not None:
                    for x in iter(in_queue.get, None):
                        if logging:
                            print('Worker %s process item: %s' % (mp.current_process(), str(x)))
                        container.append(x)
                        try:
                            fn(x, *args, **kwargs)
                        except ValueError as err:
                            print("Function %s (%s, %s) failed to process element %s" % 
                                  (fn.__name__, str(args), str(kwargs), str(x)))
                            print("Value error: {0}".format(err))
                            raise
                else:
                    for x in iter(in_queue.get, None):
                        if logging:
                            print('Worker %s process item: %s' % (mp.current_process(), str(x)))
                        try:
                            fn(x, *args, **kwargs)
                        except ValueError as err:
                            print("Function %s (%s, %s) failed to process element %s" % 
                                  (fn.__name__, str(args), str(kwargs), str(x)))
                            print("Value error: {0}".format(err))
                            raise
            else:
                if container is not None:
                    for x in iter(in_queue.get, None):
                        if logging:
                            print('Worker %s process item: %s' % (mp.current_process(), str(x)))
                        container.append(x)
                else:
                    for x in iter(in_queue.get, None):
                        if logging:
                            print('Worker %s process item: %s' % (mp.current_process(), str(x)))
                        a = x
                    warnings.warn("Items/results are not saved into container. Information lose")
            
            
            lock.acquire()
            counter.value += 1
            if logging:
                print("Finished")
                print('Worker %s assign counter to: %s' % (mp.current_process(), counter.value))
            if counter.value < n_in_process:
                lock.release()
                return
        else:
            raise ValueError("Inputs and outputs cannot be both None. ")
    
    return


class QueueChain(object):
    def __init__(self, graph, output_nodes={}):
        """ Each item in iterator graph should specify the following:
            (input_node, output_node, workers, {'fn', 'args', 'kwargs'}).
            The class will build a Queue chain to run all functions in 
            provider-consumer mode. Set up output_nodes to get the values
            out of queue. 
            About workers = 0:
            Due to the issue: https://github.com/keras-team/keras/issues/9964
            Keras/tensorflow sessions cannot be folked to mp.Process.
            So these kinds of jobs have to be run on the main process only. 
            (run pre-processor on multiple cpus, enque(batch_data) -> 
             queue.put(model.predict_batch(queue.get())) -> 
             run post-processor on multiple cpus.)
            
        Args:
            graph:
                input_node: the input_node
                output_node: the output_node
                fn: the connection function/iterator between queues
                args, kwargs: fn(*args, **kwargs)
                workers: how many workers are assigned to run this function,
                         Set up workers = 0 to use main process.
            output_nodes (Not used, specify in __run__):
                specify the queue name to get out values.
                Function will open an extra array to save values in these queues.
            
            An Example of extract nuclei features from mask rcnn result and save to image and csv
            graph = [
                (None, 'r', 0, {'fn': model.inference, 'args': (dataset,), 'kwargs': {'verbose': 0}}),
                ('r', 'f', batch_size, {'fn': extract_feature},),
                ('f', None, 1, {'fn': export_result, 'kwargs': {'save_dir': './result'}}),
            ]
            wrapper = QueueChain(graph, output_nodes=['f'])
            wrapper.run()
        """
        self.manager = mp.Manager()
        self.nodes = {None: None}
        self.graph = graph
        self.ncore = 1
        
        for in_node_id, out_node_id, workers, fn in self.graph:
            if in_node_id is not None:
                self.nodes[in_node_id][1] = workers
            if out_node_id is not None:
                self.nodes[out_node_id] = [self.manager.Queue(), 1, self.manager.Value('i', 0), self.manager.Lock(), out_node_id]
            self.ncore += workers
        assert self.ncore <= mp.cpu_count(), "Whole process requires %d cpu, but only %s observed. " % (self.ncore, mp.cpu_count())
        
    def __call__(self, output_nodes={}, logging=False):
        """ Run in parallel and save outputs"""
        ## Store result from nodes
        containers = dict([(k, self.manager.list([])) for k in output_nodes])
        
        ## Run the process
        start_time = time.time()
        main_streams = []
        p = mp.Pool(self.ncore)
        for in_node_id, out_node_id, workers, fn in self.graph:
            inputs = self.nodes[in_node_id]
            outputs = self.nodes[out_node_id]
            container = containers[in_node_id] if in_node_id in output_nodes else None
            
            if workers == 0:
                main_streams.append([inputs, outputs, fn, container])
            else:
                for _ in range(workers):
                    p.apply_async(pc_link, args=(inputs, outputs, fn, container, logging,))
        
        for inputs, outputs, fn, container in main_streams:
            pc_link(inputs, outputs, fn, container, logging)
        
        p.close()
        p.join()
        
        end_time = time.time()
        print('handle time:%ss' % (end_time - start_time))
        
        return dict([(k, list(v)) for k, v in containers.items()])


## The following functions may not needed
def softmax(logits):
    probs = np.exp(logits - np.amax(logits, axis=-1, keepdims=True))
    return probs/np.sum(probs, axis=-1, keepdims=True)


## Function to calculate IoU scores
def get_iou(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    nb_true, nb_pred = len(np.unique(y_true)), len(np.unique(y_pred))
    print("True-Class vs. Predicted-Class:", [nb_true-1, nb_pred-1])
    
    # Calculate A_and_B, A_or_B and iou_score
    A_and_B = np.histogram2d(y_true, y_pred, bins=(nb_true, nb_pred))[0]
    A_or_B = (np.histogram(y_true, bins = nb_true)[0].reshape((-1, 1))
              + np.histogram(y_pred, bins = nb_pred)[0]
              - A_and_B) + 1e-8
    return (A_and_B / A_or_B)[1:, 1:]


def get_confusion_matrix(iou_scores, threshold):
    matches = iou_scores > threshold
    true, pred = np.any(matches, axis=1), np.any(matches, axis=0)
    ## mat = [[tp, fn], [(tp), fp]] precision = mat[0,0]/(sum(mat)-mat[0,0])
    matrix = [[np.where(true)[0]+1, np.where(np.logical_not(true))[0]+1], 
              [np.where(pred)[0]+1, np.where(np.logical_not(pred))[0]+1]]
    tp, fn, tn, fp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    p = len(tp) / (len(tp) + len(fp) + len(fn))
    # print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(threshold, len(tp), len(fp), len(fn), p))
    return p, matrix


def analyze_masks_error(matrix, y_true, y_pred):
    tp, fn, tn, fp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    base_img = np.zeros(y_true.shape)
    img_predicted = base_img + sum([np.where(y_true==x, 50, 0) for x in tp] + [np.where(y_pred==x, 100, 0) for x in tn])
    img_unpredicted = base_img + sum([np.where(y_true==x, 50, 0) for x in fn] + [np.where(y_pred==x, 100, 0) for x in fp])
    return img_predicted, img_unpredicted

