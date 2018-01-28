# The code is a modification of https://github.com/QuantumLiu/tf_gpu_manager.

import os
import warnings


class GPUManagerHelper(object):
    def __init__(self):
        self.avalible = self._check_gpus()

    def _check_gpus(self):
        if not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
            warnings.warn("'nvidia-smi' tool not found.",RuntimeWarning)
            return False
        return True

    def _parse(self, line, qargs):
        numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']
        power_manage_enable = lambda v: (not 'Not Support' in v)
        to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').replace('W', ''))
        process = lambda k, v: (
            (int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
        return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}

    def by_power(d):
        '''
        helper function fo sorting gpus by power
        '''
        power_infos = (d['power.draw'], d['power.limit'])
        if any(v == 1 for v in power_infos):
            # print('Power management unable for GPU {}'.format(d['index']))
            return 1
        return float(d['power.draw']) / d['power.limit']

    def _query_gpu(self,qargs=[]):
        qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']+ qargs
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        return [self._parse(line,qargs) for line in results]

    def _sort_by_memory(self, gpus, by_size=False):
        if by_size:
            # print('Sorted by free memory size')
            return sorted(gpus, key=lambda d: d['memory.free'], reverse=True)
        else:
            # print('Sorted by free memory rate')
            return sorted(gpus, key=lambda d: float(d['memory.free']) / d['memory.total'], reverse=True)

    def _sort_by_power(self, gpus):
        return sorted(gpus, key=self.by_power)

    def _sort_by_custom(self, gpus, key, reverse=False, qargs=[]):
        if isinstance(key, str) and (key in qargs):
            return sorted(gpus, key=lambda d: d[key], reverse=reverse)
        if isinstance(key, type(lambda a: a)):
            return sorted(gpus, key=key, reverse=reverse)
        raise ValueError(
            "The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

    def auto_chooce(self, n_gpus=1, mode="by_memory"):
        # if not self.avalible:
        #     return
        gpus_info = self._query_gpu()
        avalible_modes = ["by_memory","by_memory_rate","by_power"]

        if mode=="by_memory":
            gpus_info = self._sort_by_memory(gpus=gpus_info,by_size=True)
        elif mode == "by_memory_rate":
            gpus_info = self._sort_by_memory(gpus=gpus_info, by_size=False)
        elif mode == "by_power":
            gpus_info = self._sort_by_power(gpus_info)

        choices = list(map(lambda item:item["index"],gpus_info))[:n_gpus]
        print("GPUManager select GPU{} automatically.".format(choices))
        choices = [str(item) for item in choices]
        os.environ["CUDA_VISIBLE_DEVICES"]=",".join(choices)


GPUManager = GPUManagerHelper()

