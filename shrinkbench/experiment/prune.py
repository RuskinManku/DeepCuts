import json

from .train import TrainingExperiment

from .. import strategies
from ..metrics import model_size, flops
from ..util import printc


class PruningExperiment(TrainingExperiment):

    def __init__(self,
                 dataset,
                 model,
                 strategy,
                 compression,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=True,
                 resume=None,
                 resume_optim=False,
                 save_freq=10,
                 is_LTH=False):

        super(PruningExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, resume, resume_optim, save_freq,is_LTH)
        self.add_params(strategy=strategy, compression=compression)
        self.dataset_name=dataset
        self.prune_strategy=strategy
        self.prune_compression=compression
        if is_LTH:
            self.apply_pruning(strategy, compression=1)
        else:
            self.apply_pruning(strategy, compression)
        self.is_LTH=is_LTH
        

        self.path = path
        self.save_freq = save_freq

    def apply_pruning(self, strategy, compression,is_LTH=False,init_path_LTH=None):
        print(f"I AM HERE WITH {compression}")
        constructor = getattr(strategies, strategy)
        initialized = False
        # print(self.train_dl)
        iterator = iter(self.train_dl)
        x,y = next(iterator, (None, None))
        params_ = {'compression': self.prune_compression, 'strategy': self.prune_strategy}
        self.pruning = constructor(self.model, x, y, compression=compression,is_LTH=is_LTH,init_path_LTH=init_path_LTH,strategy=self.prune_strategy)
        i=0
        while True:
            i+=1
            print(i)

            x2,y2 = next(iterator, (None, None))
            to_steps=2
            if strategy in ('LayerSmoothGrad','LayerSmoothGradCAM'):
                to_steps=2
            if x2!=None and i<to_steps and strategy in ('LayerGradCAM','GlobalGradCAMShift','GlobalGradCAM','LayerSmoothGrad','LayerSmoothGradCAM','LayerGradCAMShift'):
                self.pruning.inputs = x
                self.pruning.outputs = y
                self.pruning.apply(make_mask = False, next_iter = True)
            else:
                self.pruning.inputs = x
                self.pruning.outputs = y
                self.pruning.apply(make_mask = True, next_iter = True)
                break

            if compression==1:
                break
            x,y = x2,y2

        printc("Masked model", color='GREEN')

    def run(self):
        print("Running bitch")
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)

        self.save_metrics()

        # if self.pruning.compression > 1:
        self.epochs=1
        self.run_epochs()
        self.epochs=1
        if self.is_LTH:
            print("Now pruning and returning model to initial state, for running again")
            if self.initial_state is None:
                print("Error: Initial state not found")
                exit()
            self.apply_pruning(self.prune_strategy,self.prune_compression,self.is_LTH,self.initial_state)
            self.save_metrics()
            self.run_epochs()


    def save_metrics(self):
        self.metrics = self.pruning_metrics()
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        printc(json.dumps(self.metrics, indent=4), color='GRASS')
        summary = self.pruning.summary()
        summary_path = self.path / 'masks_summary.csv'
        summary.to_csv(summary_path)
        print(summary)

    def pruning_metrics(self):

        metrics = {}
        # Model Size
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        x, y = next(iter(self.val_dl))
        if self.dataset_name in ["SST2DATA", 'STSBDATA']:
            pass
        else:
            x, y = x.to(self.device), y.to(self.device)

        # FLOPS
            ops, ops_nz = flops(self.model, x)
            metrics['flops'] = ops
            metrics['flops_nz'] = ops_nz
            metrics['theoretical_speedup'] = ops / ops_nz

        # Accuracy
        loss, acc1, acc5 = self.run_epoch(False, -1)
        self.log_epoch(-1)

        metrics['loss'] = loss
        metrics['val_acc1'] = acc1
        metrics['val_acc5'] = acc5

        return metrics
