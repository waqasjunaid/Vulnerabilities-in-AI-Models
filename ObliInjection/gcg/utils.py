import torch
import functools
import gc
import inspect
from itertools import islice

def mellowmax(t: torch.Tensor, alpha=1.0, dim=-1):
   return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)))

def should_reduce_batch_size(exception: Exception) -> bool:
   """
   Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

   Args:
      exception (`Exception`):
         An exception
   """
   _statements = [
      "CUDA out of memory.",  # CUDA OOM
      "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
      "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
   ]
   if isinstance(exception, RuntimeError) and len(exception.args) == 1:
      return any(err in exception.args[0] for err in _statements)
   return False

def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
   """
   A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
   CUDNN, the batch size is cut in half and passed to `function`

   `function` must take in a `batch_size` parameter as its first argument.

   Args:
      function (`callable`, *optional*):
         A function to wrap
      starting_batch_size (`int`, *optional*):
         The batch size to try and fit into memory

   Example:

   ```python
   >>> from utils import find_executable_batch_size


   >>> @find_executable_batch_size(starting_batch_size=128)
   ... def train(batch_size, model, optimizer):
   ...     ...


   >>> train(model, optimizer)
   ```
   """
   if function is None:
      return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

   batch_size = starting_batch_size

   def decorator(*args, **kwargs):
      nonlocal batch_size
      gc.collect()
      torch.cuda.empty_cache()
      params = list(inspect.signature(function).parameters.keys())
      # Guard against user error
      if len(params) < (len(args) + 1):
         arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
         raise TypeError(
               f"Batch size was passed into `{function.__name__}` as the first argument when called."
               f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
         )
      while True:
         if batch_size == 0:
               raise RuntimeError("No executable batch size found, reached zero.")
         try:
               return function(batch_size, *args, **kwargs)
         except Exception as e:
               if should_reduce_batch_size(e):
                  gc.collect()
                  torch.cuda.empty_cache()
                  batch_size //= 2
                  print(f"Decreasing batch size to: {batch_size}")
               else:
                  raise

   return decorator

def batched(iterable, n):
   it = iter(iterable)
   while batch := list(islice(it, n)):
      yield batch

class PerplexityFilter:
    """
    The implementation is adapted from https://github.com/neelsjain/baseline-defenses/blob/main/perplexity_filter.py

    """
    def __init__(self, model, tokenizer, threshold, window_size='all'):
        self.tokenizer = tokenizer
        self.model = model
        self.threshold = threshold
        self.window_threshold = threshold
        self.window_size = window_size
        self.cn_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def detect(self, sequence):
        if self.window_size == 'all':
            return (not self.filter([sequence])[-1][0])
        elif self.window_size <= 0 or type(self.window_size) != int:
            raise ValueError(f"ERROR: window_size={self.window_size}. window_size must be a positive integer.")
        return (not self.filter_window([sequence])[-1][0])
    
    def get_log_prob(self, sequence):
        input_ids = self.tokenizer.encode(sequence, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            # logits = self.model(input_ids, return_logits_with_embedding=True)[0]    # For PaLM
            logits = self.model(input_ids, labels=input_ids).logits   # For other models, e.g., Vicuna
        logits = logits[:, :-1, :].contiguous()
        input_ids = input_ids[:, 1:].contiguous()
        log_probs = self.cn_loss(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        return log_probs
    
    def filter(self, sequences):
        filtered_log_ppl = []
        passed_filter = []
        for sequence in sequences:
            log_probs = self.get_log_prob(sequence)
            NLL_by_token = log_probs
            if NLL_by_token.mean() <= self.threshold:
                passed_filter.append(True)
                filtered_log_ppl.append(NLL_by_token.mean().item())
            else:
                passed_filter.append(False)
                filtered_log_ppl.append(NLL_by_token.mean().item())
        print(filtered_log_ppl, passed_filter)
        return filtered_log_ppl, passed_filter
    
    def filter_window(self, sequences, reverse=False):
        filtered_log_ppl_by_window = []
        passed_filter_by_window = []
        passed = []
        for sequence in sequences:
            sequence_window_scores = []
            passed_window_filter = []
            log_probs = self.get_log_prob(sequence)
            NLL_by_token = log_probs
            for i in np.arange(0, len(NLL_by_token), self.window_size):
                if not reverse:
                    window = NLL_by_token[i:i+self.window_size]
                else:
                    if i == 0:
                        window = NLL_by_token[-self.window_size:]
                    elif -(-i-self.window_size) > len(NLL_by_token) and i != 0:
                        window = NLL_by_token[:-i]
                    else:
                        window = NLL_by_token[-i-self.window_size:-i]
                if window.mean() <= self.window_threshold:
                    passed_window_filter.append(True)
                    sequence_window_scores.append(window.mean().item())
                else:
                    passed_window_filter.append(False)
                    sequence_window_scores.append(window.mean().item())
            if all(passed_window_filter):
                passed.append(True)
            else:
                passed.append(False)
            passed_filter_by_window.append(passed_window_filter)
            filtered_log_ppl_by_window.append(sequence_window_scores)
        return filtered_log_ppl_by_window, passed_filter_by_window, passed
