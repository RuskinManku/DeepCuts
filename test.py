from shrinkbench.experiment import PruningExperiment


for strategy in ['GlobalMagWeight']:
    for  c in [1.25]:
        exp = PruningExperiment(dataset='SST2DATA', 
                                model='BertNet',
                                strategy=strategy,
                                compression=c,
                                train_kwargs={'epochs':6},
                                is_LTH=True)
        exp.run()
