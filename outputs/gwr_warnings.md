# With any ethnicity dummy variable when kernel = bisquare

---

\_RemoteTraceback Traceback (most recent call last)
\_RemoteTraceback:
"""
Traceback (most recent call last):
File "c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\joblib\externals\loky\process_executor.py", line 490, in \_process_worker
r = call_item()
File "c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\joblib\externals\loky\process_executor.py", line 291, in **call**
return self.fn(\*self.args, \*\*self.kwargs)

```^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\joblib\parallel.py", line 607, in **call**
return [func(*args, **kwargs) for func, args, kwargs in self.items]
~~~~^^^^^^^^^^^^^^^^^
File "c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\mgwr\gwr.py", line 267, in \_local_fit
betas, inv_xtx_xt = \_compute_betas_gwr(self.y, self.X, wi)
~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
File "c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\spglm\iwls.py", line 37, in \_compute_betas_gwr
xtx_inv_xt = linalg.solve(xtx, xT)
File "c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\scipy_lib_util.py", line 1233, in wrapper
return f(*arrays, *other_args, **kwargs)
File "c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\scipy\linalg_basic.py", line 297, in solve
\_solve_check(n, info)
~~~~~~~~~~~~^^^^^^^^^
File "c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\scipy\linalg_basic.py", line 43, in \_solve_check
raise LinAlgError('Matrix is singular.')
numpy.linalg.LinAlgError: Matrix is singular.
"""

The above exception was the direct cause of the following exception:

LinAlgError Traceback (most recent call last)
Cell In[46], line 32
24 model = GWR(
25 train_coords,
26 train_target,
27 train_predictors,
28 bw = 200
29 )
31 # Fit model
---> 32 results = model.predict(
33 val_coords, val_predictors
34 )
36 # Get predictions
37 predictions = results.predy

File c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\mgwr\gwr.py:397, in GWR.predict(self, points, P, exog_scale, exog_resid, fit_params)
372 """
373 Method that predicts values of the dependent variable at un-sampled
374 locations
(...) 394
395 """
396 if (exog_scale is None) & (exog_resid is None):
--> 397 train_gwr = self.fit(\*\*fit_params)
398 self.exog_scale = train_gwr.scale
399 self.exog_resid = train_gwr.resid_response

File c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\mgwr\gwr.py:348, in GWR.fit(self, ini_params, tol, max_iter, solve, lite, pool)
345 else:
346 m = self.points.shape[0]
--> 348 rslt = Parallel(n_jobs=self.n_jobs)(delayed(self.\_local_fit)(i) for i in range(m))
350 rslt_list = list(zip(\*rslt))
351 influ = np.array(rslt_list[0]).reshape(-1, 1)

File c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\joblib\parallel.py:2072, in Parallel.**call**(self, iterable)
2066 # The first item from the output is blank, but it makes the interpreter
2067 # progress until it enters the Try/Except block of the generator and
2068 # reaches the first `yield` statement. This starts the asynchronous
2069 # dispatch of the tasks to the workers.
2070 next(output)
-> 2072 return output if self.return_generator else list(output)

File c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\joblib\parallel.py:1682, in Parallel.\_get_outputs(self, iterator, pre_dispatch)
1679 yield
1681 with self.\_backend.retrieval_context():
-> 1682 yield from self.\_retrieve()
1684 except GeneratorExit:
1685 # The generator has been garbage collected before being fully
1686 # consumed. This aborts the remaining tasks if possible and warn
1687 # the user if necessary.
1688 self.\_exception = True

File c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\joblib\parallel.py:1784, in Parallel.\_retrieve(self)
1778 while self.\_wait_retrieval():
1779 # If the callback thread of a worker has signaled that its task
1780 # triggered an exception, or if the retrieval loop has raised an
1781 # exception (e.g. `GeneratorExit`), exit the loop and surface the
1782 # worker traceback.
1783 if self.\_aborting:
-> 1784 self.\_raise_error_fast()
1785 break
1787 nb_jobs = len(self.\_jobs)

File c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\joblib\parallel.py:1859, in Parallel.\_raise_error_fast(self)
1855 # If this error job exists, immediately raise the error by
1856 # calling get_result. This job might not exists if abort has been
1857 # called directly or if the generator is gc'ed.
1858 if error_job is not None:
-> 1859 error_job.get_result(self.timeout)

File c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\joblib\parallel.py:758, in BatchCompletionCallBack.get_result(self, timeout)
752 backend = self.parallel.\_backend
754 if backend.supports_retrieve_callback:
755 # We assume that the result has already been retrieved by the
756 # callback thread, and is stored internally. It's just waiting to
757 # be returned.
--> 758 return self.\_return_or_raise()
760 # For other backends, the main thread needs to run the retrieval step.
761 try:

File c:\Users\Natha\Dev\irp\.venv\Lib\site-packages\joblib\parallel.py:773, in BatchCompletionCallBack.\_return_or_raise(self)
771 try:
772 if self.status == TASK_ERROR:
--> 773 raise self.\_result
774 return self.\_result
775 finally:

LinAlgError: Matrix is singular.
```
