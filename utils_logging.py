import numpy as np
import json
import wandb
from accelerate.tracking import GeneralTracker, on_main_process


def get_mean_std_max_min_dict(array, prefix):
    res = {}
    res[prefix + '/mean'] = np.mean(array)
    res[prefix + '/std'] = np.std(array)
    res[prefix + '/min'] = np.amin(array)
    res[prefix + '/max'] = np.amax(array)
    return res


class Metrics():

    def __init__(self, *args, log_with_wandb=False):
        self.log_with_wandb = log_with_wandb
        self.metrics = {arg: 0 for arg in args}
        self.latest_metrics = {arg: 0 for arg in args}
        self.samples = {arg: 1e-8 for arg in args}
        self.logged_metrics = [arg for arg in args]

    def reset(self,):
        for arg in self.metrics:
            self.metrics[arg] = 0
            self.samples[arg] = 1e-8

    def add(self, *args):
        for arg in args:
            if arg not in self.metrics:
                self.logged_metrics.append(arg)
                self.metrics[arg] = 0
                self.latest_metrics[arg] = 0
                self.samples[arg] = 1e-8

    def update(self, **kwargs):
        for arg, val in kwargs.items():
            if arg not in self.metrics:
                self.logged_metrics += arg
                self.metrics[arg] = 0
                self.latest_metrics[arg] = 0
                self.samples[arg] = 1e-8
            self.metrics[arg] += val
            self.samples[arg] += 1

    def set(self, **kwargs):
        for arg, val in kwargs.items():
            if arg not in self.metrics:
                self.logged_metrics += arg
                self.metrics[arg] = val
                self.samples[arg] = 1
            self.metrics[arg] = val
            self.samples[arg] = 1

    def get(self,):
        for arg, metric_agg in self.metrics.items():
            samples = self.samples[arg]
            if samples >= 1:
                self.latest_metrics[arg] = metric_agg/samples
        return self.latest_metrics

    def log(self, logging_file, reset=True, keys_to_print=[]):
        latest_metrics = self.get()
        if self.log_with_wandb:
            wandb.log(latest_metrics)
        with open(f'{logging_file}', 'a') as f:
            json_data = json.dumps(latest_metrics, indent=4)
            f.write(json_data)
            f.write('\n')
        if len(keys_to_print) > 0:
            print(f'----')
            for key in keys_to_print:
                if key in latest_metrics:
                    print(f'{key}: {latest_metrics[key]}')
            print(f'----')
        if reset:
            self.reset()
        return latest_metrics
